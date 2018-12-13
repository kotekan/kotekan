************
Style Guide
************

If you're going to work on **kotekan** code, please adhere to the following guidelines:


General Rules
-------------
* Use 4-space tabs. (**Not** Tab characters!)

* No trailing whitespace.

* Files should end with a newline.


Detailed Rules
---------------
.. toctree::
    :maxdepth: 1

    dev_style_guide_doc
    dev_style_guide_config
    dev_style_guide_code



File Organization
------------------

* ``kotekanProcess`` files should be placed in the **lib/processes/** folder, and named in *camelCase* to match the process class they contain.

* Accelerator interfaces (such as e.g. OpenCL or CUDA) should create their own folder within **lib/**, for storage of kernel handling objects and the kernels themselves.

* Support scripts and components can be placed in the **script/** folder.

* Configuration (``yaml``) files should go in the **config/** folder.

