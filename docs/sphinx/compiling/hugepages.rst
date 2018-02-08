..  _hugepages:

=======================
Configuring Hugepages
=======================

(Instructions for Linux hosts with hugepage support.)

Add the following line to /etc/fstab to enable huge pages

.. code:: bash

    nodev   /mnt/huge   hugetlbfs   defaults        0 0

And make the folder /mnt/huge

.. code:: bash

    sudo mkdir /mnt/huge
