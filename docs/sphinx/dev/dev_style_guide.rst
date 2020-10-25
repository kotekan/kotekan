************
Style Guide
************

If you're going to work on **kotekan** code, please adhere to the following
guidelines:


General Rules
-------------
* Use 4-space tabs. (**Not** Tab characters!)

* No trailing whitespace.

* Lines should not exceed 80 columns.

* Files should end with a newline.


Detailed Rules
---------------
.. toctree::
    :maxdepth: 1

    dev_style_guide_doc
    dev_style_guide_config
    dev_style_guide_code
    dev_style_guide_cmake


File Organization
------------------

* ``kotekan::Stage`` files should be placed in the **lib/processes/** folder,
  and named in *CamelCase* to match the stage class they contain.

* Accelerator interfaces (such as e.g. OpenCL or CUDA) should create their own
  folder within **lib/**, for storage of kernel handling objects and the kernels
  themselves.

* Support scripts can be placed in the **script/** directory.

* Python components should be placed in the **python/** directory.

* Configuration (``yaml``) files should go in the **config/** folder.
