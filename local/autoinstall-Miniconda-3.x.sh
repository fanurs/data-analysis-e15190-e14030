#!/bin/bash
#
# This script downloads and installs the latest Miniconda-3.x
#
#
#
#### REPLACE THE FOLLOWING LINK WITH THE MINICONDA VERSION THAT YOU WANT TO INSTALL ####
#### SEE https://docs.conda.io.en/latest/miniconda.html FOR MORE INFORMATION        ####
web_source='https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh'

# download the latest Miniconda-3.x installer
download_path=$(basename -- $web_source)
echo "> Begin to download from: ${web_source}"
wget $web_source -O $download_path
echo "> Download done."

# run the installer
this_dir=`dirname "$(readlink -f "$0")"`
response="\n"
response="${response}yes\n"
response="${response}${this_dir}/miniconda3-latest\n"
response="${response}no\n"
echo -e $response | /bin/bash ${download_path}
