********************
Theory of Operation
********************

This page explains how a kotekan pipeline is put together at runtime, with the
emphasis on the GPU infrastructure: how a ``cudaProcess`` stage drives a list of
commands, how frames flow through the device, and the rules a command author or
config author has to follow. Most of these rules are consequences of the design
rather than checks in the code, so they are collected here. The host-side buffer
protocol is covered in :ref:`dev_buffers`; this page assumes you have read it.

Host pipeline in one paragraph
==============================

A pipeline is a set of stages connected by buffers. Each stage runs its own
``main_thread``, and may start more threads of its own. A
producer waits for an empty frame, fills it, and marks it full; a consumer waits
for a full frame, reads it, and marks it empty. Every frame can carry one
metadata object drawn from a pool. Stages never signal each other directly; all
coordination goes through the buffers. A GPU pipeline keeps this model on the
host side and adds a second, asynchronous world on the device.


The GPU stage
=============

A ``cudaProcess`` is an ordinary stage: it registers as a consumer of its input
buffers and as a producer of its output buffers, and runs in its own threads. What
makes it different is that it does not process data itself. It owns an ordered
list of **commands**, and each command does one thing on the device: copy a host
frame in, run a kernel, wait for a stream, or copy a result out. The stage runs
the whole list once per **GPU frame**.

A minimal config looks like this:

.. code-block:: yaml

    buffer_depth: 4
    frame_arrival_period: 5.12e-6 * samples_per_data_set

    host_voltage_buffer:
        kotekan_buffer: ndarray
        num_frames: buffer_depth
        ...
    host_correlation_buffer:
        kotekan_buffer: ndarray
        num_frames: buffer_depth
        ...

    corr:
        gpu_0:
            kotekan_stage: cudaProcess
            gpu_id: 0
            in_buffers:
                voltage_in: host_voltage_buffer
            out_buffers:
                corr_out: host_correlation_buffer
            commands:
                - name: cudaInputData
                  in_buf: voltage_in
                  gpu_mem: voltage_buffer
                - name: cudaSyncInput
                - name: cudaCorrelator
                  ...
                - name: cudaSyncOutput
                - name: cudaOutputData
                  gpu_mem: correlation_buffer
                  out_buf: corr_out

Three things to notice:

- ``in_buffers`` and ``out_buffers`` map a local name to a global buffer. Commands
  refer to buffers by the local name. At startup the stage page-locks every
  frame of every host buffer it is given, so those buffers cost pinned memory.
- ``gpu_mem`` names are **device** memory regions, not host buffers. They exist only
  inside the device interface and are created the first time a command asks for
  them (see `GPU memory`_).
- ``buffer_depth`` is the single most important parameter. It sets how many GPU
  frames can be in flight at once, how many instances of each command exist, and
  how many copies of every per-frame device array are allocated. Host buffers
  feeding a GPU stage are normally declared with ``num_frames: buffer_depth`` so
  that the two ring lengths agree.
- Besides ``gpu_id`` and ``commands``, the stage requires ``buffer_depth``,
  ``log_level``, ``cpu_affinity`` and, unless ``profiling: false`` is set,
  ``frame_arrival_period``. These are usually inherited from the top level of the
  config.


Frame life cycle
================

``gpuProcess`` runs two threads. The stage's own ``main_thread`` *queues* work; a
``results_thread`` *retires* it. Both count GPU frames from zero with the same
counter, ``gpu_frame_id``, and both index the per-frame slot with
``gpu_frame_id % buffer_depth``.

For each GPU frame the main thread does, in order:

1. **Wait for a free slot.** Slot ``gpu_frame_id % buffer_depth`` must have been
   retired by the results thread. This is what bounds the number of frames in
   flight to ``buffer_depth``.
2. **``start_frame(gpu_frame_id)``** on the slot's instance of every command. This
   just records the frame counter.
3. **``wait_on_precondition()``** on every command, in list order. This is the only
   place a command may block. Copy-in commands wait for a full host frame;
   copy-out commands wait for an empty one; ring buffer producers and consumers
   claim their region here. A return of ``-1`` means shutdown and ends the loop.
4. **``execute()``** on every command, in list order, under the device's command
   mutex. Each command enqueues its copy or kernel launch on a CUDA stream and
   returns an event recorded after it; nothing should wait for the device here.
5. **Hand the last event to the slot's signal** and loop.

The results thread, for each frame in the same order:

1. **Wait on the slot's signal**, i.e. ``cudaEventSynchronize`` on the event the last
   command returned. When it fires, the GPU has finished that frame.
2. **``finalize_frame()``** on every command, in list order. This is where a command
   releases what it claimed in its precondition: mark the host input frame empty,
   mark the host output frame full, finish a ring buffer read or write, record
   profiling times.
3. **Reset the slot**, which lets the main thread reuse it for frame
   ``gpu_frame_id + buffer_depth``.

Consequences for command authors
--------------------------------

- **Block only in ``wait_on_precondition``.** ``execute`` runs while the command
  mutex is held and every later command in this frame, and every frame after
  it, waits for it to return. A host-side ``cudaDeviceSynchronize``, a blocking
  buffer wait, or a slow host computation in ``execute`` stalls the whole device.
- **Release only in ``finalize_frame``.** Marking an input frame empty in ``execute``
  would let the producer overwrite it while the asynchronous copy is still
  reading it. The same holds for ring buffer regions.
- **``finalize_frame`` is not called during shutdown.** Do not rely on it to free
  memory or restore invariants that matter after the stage stops.
- **Preconditions and finalizations run in frame order in a single thread each.**
  A command may rely on the fact that its precondition for frame ``n`` runs after
  the precondition of every earlier frame and every earlier command in the list.
  This is what makes per-frame ring buffer claims consistent.


Instances
=========

A command class is instantiated ``buffer_depth`` times per list entry. Instance
``i`` handles every frame with ``gpu_frame_id % buffer_depth == i``, so its member
variables are **per-slot state**, not per-command state. This has several
non-obvious effects:

- **Register producers and consumers from instance 0 only.** A buffer identifies
  a producer or consumer by name, and all instances share the command's unique
  name. Every command in the tree guards its registration with
  ``if (instance_num == 0)``. A second registration under the same name is an
  error on a frame buffer and an exception on a ring buffer.
- **A "first time" flag is per instance.** Code of the form "on the first frame,
  set up the metadata" runs once per instance, so ``buffer_depth`` times, unless
  it is also guarded by ``instance_num == 0``. The ring buffer producers use
  exactly that pair of conditions.
- **State that must be shared goes in a ``cudaCommandState``.** Register the
  command with ``REGISTER_CUDA_COMMAND_WITH_STATE`` and one state object is
  created before the instances and handed to each of them.
- **The host frame a command touches is ``gpu_frame_id % num_frames``.** With
  ``num_frames == buffer_depth`` this equals the instance number, so each
  instance always sees the same host frame. Other values work, but the
  correspondence is lost.


Streams and ordering
====================

Within one frame the commands are *enqueued* in list order, but they *run* on
the device asynchronously, and CUDA only orders work within a single stream.
``cudaProcess`` creates ``num_cuda_streams`` streams, three by default, and a
command's type picks its default stream:

============  ======  ==============================
Command type  Stream  Default users
============  ======  ==============================
``COPY_IN``   0       ``cudaInputData``, ring producers
``COPY_OUT``  1       ``cudaOutputData``, ring consumers
``KERNEL``    2       every kernel
============  ======  ==============================

A command can override this with ``cuda_stream`` in its config block; the
generic ``cudaSyncStream`` barrier has no default and must set it.

Dependencies between streams are expressed with events. ``queue_commands`` keeps
the last event recorded on each stream and passes the table to every ``execute``
as ``pre_events``. A command only waits for what it explicitly waits for, and
most commands wait for nothing beyond their own stream's order. So by default
**nothing on the kernel stream waits for the copy-in stream**, and nothing on the
copy-out stream waits for the kernels. The barrier commands supply those edges:

- ``cudaSyncInput`` runs on the kernel stream and waits for the last event on
  stream 0. Put it after the copy-ins and before the first kernel.
- ``cudaSyncOutput`` runs on the copy-out stream and waits for the last event on
  every stream numbered 2 and up. Put it after the last kernel and before the
  copy-outs.
- ``cudaSyncStream`` is the general form: it waits on a configured list of
  ``source_cuda_streams``.

The canonical command list is therefore: copy-ins, ``cudaSyncInput``, kernels,
``cudaSyncOutput``, copy-outs. A kernel that reads from a ring buffer filled by
*another* stage needs no ``cudaSyncInput``, because the ring buffer protocol
already guarantees the data landed before the region was claimable.

One more rule hides in the frame life cycle: **the slot's signal is the event of
the last command in the list**, not a join of all streams. The frame is
considered finished when that one event fires. If the last command's event does
not transitively cover the work on the other streams, the results thread will
retire the frame early and the slot and its device memory can be reused while
work is still running. Ending the list with ``cudaSyncOutput`` and a copy-out
satisfies this; ending it with a kernel on a stream that nothing waits on does
not.


GPU memory
==========

Device memory is not declared in the config. It is a flat namespace inside the
``cudaDeviceInterface``, shared by every stage on the same ``gpu_id``, and a
region is allocated the first time a command asks for it by name. There are two
kinds:

- ``get_gpu_memory(name, len)`` returns a **singleton**: one region, the same for
  every frame. Use it for lookup tables, ``do_once`` inputs, and the backing
  store of a ring buffer.
- ``get_gpu_memory_array(name, gpu_frame_id, buffer_depth, len)`` returns element
  ``gpu_frame_id % buffer_depth`` of an **array** of ``buffer_depth`` regions,
  allocated contiguously. Use it for anything that is per frame: copy-in
  targets, kernel outputs, copy-out sources.

Rules that follow:

- **The size is fixed by the first request.** A later request under the same
  name with a different ``len`` logs an error and asserts. Two commands that
  share a region by name must agree on its size exactly.
- **Names are per device, not per stage.** Two ``cudaProcess`` stages on the same
  GPU that both ask for ``voltage_buffer`` get the same memory. This is how
  stages share data on the device, and it is also how they collide by accident.
- **The reuse distance of an array element is ``buffer_depth`` frames**, which is
  exactly the number of frames that can be in flight, so an element is never
  overwritten while its frame is still being processed, provided the ordering
  rule in the previous section holds.
- **Views** (``create_gpu_memory_view``, ``create_gpu_memory_array_view``,
  ``create_gpu_memory_ringbuffer``) expose a sub-range of a region under another
  name without allocating. A view must be created before any command requests
  the view name, so it belongs in a constructor, and creating it twice throws,
  so guard it with ``instance_num == 0``.

Each array element also carries **one metadata pointer**. ``cudaInputData``
attaches the host frame's own object to the element it copies into, so the
device element and the host frame share it. A kernel gives its output element a
fresh object from the pool, deep-copied from an input and then adjusted, which
is what ``NDArrayBuffer::set_metadata`` does. ``cudaOutputData`` attaches the
object on its source element to the host output frame. The rule for a kernel is
therefore: read input metadata, never modify it, and create the output's object
per frame. ``cudaOutputData`` with an ``in_buf`` set ignores device metadata and
passes the host input frame's object straight through.


Profiling and flags
===================

Every command may call ``record_start_event`` and must return the event from
``record_end_event``. ``finalize_frame`` measures the elapsed device time between
them and feeds the ``*_execute_time`` and ``*_u`` trackers, where utilization is
time divided by ``frame_arrival_period``. The per-stage summary is served at
``/gpu_profile/<stage unique name>`` on the REST port, and ``log_profiling: true``
prints it once per frame.

A ``cudaPipelineState`` object is created per frame and passed through every
``execute``. Commands can set named flags and integers on it; a command with
``required_flag: <name>`` in its config is skipped for that frame when the flag
is not set. ``cudaOutputData`` remembers whether it was skipped so that its
``finalize_frame`` does not mark an output frame full that it never wrote.

``cudaInputData`` with ``do_once: true`` copies its host frame into a singleton
region on the first frame and then never touches the buffer again. This is the
mechanism for constant inputs such as beamforming phases.


Ring buffers
============

Frame-based buffers force every stage on the chain to agree on one frame size.
GPU stages on the same device often want to run at different cadences, or read
overlapping windows, or have one stage's output be several stages' input. The
``RingBuffer`` exists for this, and it is the part of the infrastructure with
the most unwritten rules.

What a ring buffer is
---------------------

A ``RingBuffer`` (``kotekan_buffer: ring``) owns **no data**. It is a signalling
object over a region of memory that lives elsewhere, in practice a
``get_gpu_memory`` singleton of ``ring_buffer_size`` bytes on the device. It tracks
byte cursors: for each producer, how far it has written; for each consumer, how
far it has claimed and how far it has finished reading. The data region itself is addressed
by the producer and consumer commands directly, as ``cursor % ring_buffer_size``.

By convention the three objects for a quantity ``X`` are named:

- ``host_X_buffer``: the host frame buffer feeding or draining the ring, if any;
- ``X_buffer``: the device region, i.e. the ``get_gpu_memory`` name;
- ``host_X_ringbuffer``: the ``RingBuffer`` signalling object in the config.

``NDArrayRingBuffer`` and ``NDArrayBuffer`` derive all three names from ``X``, so a
kernel that uses them must be configured with matching names.

Producer and consumer protocol
------------------------------

A producer, per frame:

1. ``wait_for_writable(name, instance, bytes)`` in ``wait_on_precondition``. Returns
   the write cursor once ``bytes`` are free, and reserves them so that the next
   call (from the next frame's instance) gets the region after it.
2. Write into the region in ``execute``, splitting the copy at the wrap point.
3. ``finish_write(name, instance, bytes)`` in ``finalize_frame``. Only now do
   consumers see the data.

A consumer, per frame:

1. ``wait_and_claim_readable(name, instance, bytes)`` in ``wait_on_precondition``.
   Returns the read cursor once ``bytes`` have been produced by every producer,
   and advances this consumer's head.
2. Read the region in ``execute``.
3. ``finish_read(name, instance, bytes)`` in ``finalize_frame``. Only now can a
   producer overwrite the region.

The reservations happen in preconditions, which run in frame order in the
stage's single main thread, so instance ``i`` of a consumer always claims the
chunk that follows the one instance ``i-1`` claimed. **A ring buffer therefore
only works with the precondition and finalize discipline described above.**
Claiming in ``execute`` would let instances race for cursors.

Other points that are not obvious from the interface:

- **Sizes are bytes, not elements.** ``ring_buffer_size``, ``input_size`` and
  ``output_size`` in the config are byte counts. ``NDArrayRingBuffer`` converts
  between elements of the slowest dimension and bytes for you.
- **Multiple producers write the same elements.** The ring's readable head is the
  minimum over producers, so several producers are expected to each fill part
  of every chunk, not to interleave chunks. Multiple consumers each see every
  byte.
- **A consumer can read more than it claims.** ``NDArrayRingBuffer::wait_and_claim_readable``
  takes a callback that decides, from the number of available elements, how many
  to *read* and how many to *claim*. The difference is the overlap that will be
  presented again next frame. This is how windowed kernels get their history
  without copying.
- **Wrap-around is the kernel's problem.** The copy commands split at the wrap
  point, but a kernel handed a pointer into the ring does not. Kernels in the
  tree assert that a chunk never straddles the end, which holds when
  ``ring_buffer_size`` is a multiple of the chunk size. The usual choice is
  ``ring_buffer_size = buffer_depth * chunk``.
- **A ring buffer with no consumer discards data**, silently, as soon as it is
  written.

Ring buffer metadata
--------------------

A frame buffer has one metadata object per frame. A ring buffer has ``num_frames
== 1``, so it has **one metadata slot, index 0, describing the whole ring**. That
object is set exactly once, by instance 0 of the producer on its first frame,
before its first ``finish_write``. It records the shape, type, dimension names,
and the FPGA sequence number of **element 0 of the ring**. It does not change
after that, and it carries no per-frame information.

Consumers derive per-frame values from the cursor: the sequence number of the
chunk starting at element ``e`` is ``fpga_seq_num + e * time_downsampling_fpga``.
``cudaCopyFromRingbuffer`` deep-copies the slot-0 object into a fresh one per
frame, adjusts the sequence number and the leading extent, and attaches that to
the host output frame; kernels do the same for their output arrays. Rules:

- **Set the slot once, from instance 0, on the first frame.** Every instance of
  the producer reads it back to learn the origin sequence number; only instance
  0 writes it. Writing it on every frame, or from every instance, replaces the
  object while other stages are reading it, and other threads read it without
  holding the buffer mutex.
- **Never modify the slot-0 object in place.** Copy it, then modify the copy.
- **Start the ring at cursor 0 on the first frame.** The producers assert this
  because the recorded sequence number is tied to element 0.
- **Check, do not assume, that inputs agree.** A kernel with two ring inputs
  should assert that their sequence numbers at the claimed cursors coincide;
  ``cudaCorrelator`` shows the pattern.

The ``NDArrayRingBuffer`` helper
--------------------------------

Kernels do not usually call the ``RingBuffer`` methods directly. They hold an
``NDArrayRingBuffer<T, D>`` per ring input or output, constructed with the
quantity name, the full extents of the ring as an array, the dimension names and
scalings, and a reference to the command. It:

- registers the producer or consumer from instance 0;
- converts the byte cursors into element extents (``get_read_valid``,
  ``get_read_claimed``, ``get_write_valid``) that index the slowest dimension;
- hands out a typed ``NDArray`` view of the device region;
- checks the slot-0 metadata against the declared shape and type, and sets it
  for output rings;
- can poison a region and check for poison, for debugging.

Its life cycle is the command's: claim in ``wait_on_precondition``, use in
``execute``, release in ``finalize_frame``, and it asserts if these are called out
of order.


Connecting stages on one device
===============================

A device pipeline is usually several ``cudaProcess`` stages, each with its own
threads, command list and ``buffer_depth``, connected by ring buffers. The
example ``config/ci-tests/gpu_batch/verify_cuda_n2k.yaml`` has six: three copy
host frames into three rings, one correlates from the voltage and RFI mask
rings, one expands the packet-loss mask from its ring into another ring, and
one correlates from that expanded ring and the RFI mask ring. Because the rings
decouple cadences, the mask stages run with a different chunk size from the
voltage stage, and each correlator waits until both of its inputs have enough
data.

The reason to split a device pipeline into stages rather than one long command
list is that each stage blocks independently. A single list stalls at its
slowest precondition every frame; separate stages let a fast producer run ahead
by up to a ring's worth of data.


Checklist for a new command
===========================

- Derive from ``cudaCommand``, register with ``REGISTER_CUDA_COMMAND`` or the
  ``_WITH_STATE`` variant, and call ``set_command_type`` in the constructor so a
  stream is assigned.
- Register host buffer producers and consumers, and ring producers and
  consumers, from instance 0 only.
- Block in ``wait_on_precondition`` only; return ``-1`` on shutdown.
- In ``execute``: call ``pre_execute``, fetch device memory by name with the same
  size every time, ``record_start_event``, enqueue on ``device.getStream(cuda_stream_id)``,
  set output metadata, and return ``record_end_event()``. No host-side waits.
- In ``finalize_frame``: release frames and ring regions, then call
  ``cudaCommand::finalize_frame()``.
- List the device regions the command reads and writes in ``gpu_buffers_used``,
  or use the ``NDArray*`` helpers which do it, so the pipeline viewer can draw it.
- If the command reads a ring, decide how much to claim versus read, and derive
  per-frame sequence numbers from the cursor.
- If the command is last in a list, make sure its end event covers every stream
  the list used.
