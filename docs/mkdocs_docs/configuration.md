Configuration
=============

RolyPoly uses a configuration file (rpconfig.json) to store settings
such as the data directory. This file is automatically updated when
using the prepare-external-data command.

To view or modify the configuration:

1.  Locate the rpconfig.json file in the root directory of the RolyPoly
    project.
2.  Open the file with a text editor.
3.  Modify the settings as needed.
4.  Save the file.

Example configuration:

    {
            "ROLYPOLY_DATA": "/REDACTED_HPC_PATH/projects/rolypoly/data/"
    }

Note: It's recommended to use the prepare-external-data command to set
up the initial configuration and download necessary resources.
