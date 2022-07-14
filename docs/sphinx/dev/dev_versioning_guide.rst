****************
Versioning Guide
****************

Generally all development work leading to a new release goes into the ``develop`` branch
(using a feature branch and pull-request model).
Once the ``develop`` branch is ready for a release
(usually because an instrument is starting a science data run),
it gets merged into ``main``, at which point it is tagged according to the version format guidelines below:

The version tagging format follows this style:

``YYYY.MM[.N][revision]``.

Where ``YYYY`` is the year, ``MM`` is the month with a leading zero, ``N`` is used if more than one full release
happens in that month and starts at 1 (with zero for the first release implied).
The ``revision`` is for a hot fix applied to a specific release, and is a letter, starting with ``'a'``.

For example the first release in Mar of 2022 would be ``2022.03``, the second would be ``2022.03.1``,
and a hot fix to ``2022.03`` would be ``2022.03a``.

Full release tags should always be made against the ``main`` branch.  For hot fix releases, a new
branch with the name ``[version]_hotfix`` should be created and used for all hot fixes.  So in the example
above ``2022.03`` would get a branch named ``2022.03_hotfix`` and tags on that branch of ``2022.03a``,
``2022.03b``, etc.   All hot fixes should pushed both to hot fix branch *and* ``develop`` so they
are merged into all future full releases.


Generally releases will occur when a telescope or subsystem using ``kotekan`` goes into an science observing
mode requiring changes currently only in the ``develop`` branch.  For example if telescope ``X`` needs
feature ``A`` in ``develop``, then all of ``develop`` will be merged into ``main`` and a release will be generated.
This means that features in ``develop`` for any system should not be incomplete, in the sense that
they cannot break existing functionality pending another unmerged change, and also that the feature ``A`` shouldn't
break the functionality of telescope ``Y``.
That is partly enforced by requiring all unit tests to pass priory to merging feature branches into ``develop``, and
by careful code reviews.

Once an instrument is running a given version, if a problem is found that only impacts that instrument,
then instead of moving to the latest version, they can opt to generate a hot fix release.
For example if telescope X is running version ``2022.03`` and finds a bug, but wants to stay
on that version for code stability during a science run, then the bug can be fixed with a hot fix
release.

The goal of this design is to maintain only one primary branch which all instruments can use, and that instruments
will generally strive to run the latest version available.  While also allowing for instruments to opt to stay
on a given release version for an extended period of time, where code stability is more desirable than being upto date.
However, using hot fix branches to add new features as "instrument specific branches" is strongly discouraged, as
this will lead to code fragmentation and eventually prevent older systems from making use of features in newer versions
of the framework.
All instruments should strive to be upto date with the latest code versions on some cadence, even if it is fairly slow
(e.g. yearly for well established systems).  Developers for new instruments should take care
not to break existing functionality, or if needed, do so in a well documented way giving the older systems a path to
upgrade.  For example an API could change its signature and support new features, but must still provide the original
functionality with minimal changes for the older instrument to update.