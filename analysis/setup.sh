#!/usr/bin/env bash

action() {
    #
    # local variables
    #

    # FIXME
    #source "/nfs/dust/cms/user/frengelk/Anaconda/etc/profile.d/conda.sh"
    #conda activate

    local this_file="$( [ ! -z "$ZSH_VERSION" ] && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "$this_file" )" && pwd )"

    local vpython="$( python -c "import sys; print('{0.major}.{0.minor}'.format(sys.version_info))" )"


    #
    # global variables
    #

    export ANALYSIS_BASE="$this_dir"
    export ANALYSIS_STORE="$ANALYSIS_BASE/tmp/data"
    export ANALYSIS_SOFTWARE="$ANALYSIS_BASE/tmp/software"
    export ANALYSIS_PATH=/nfs/dust/cms/group/susy-desy/Susy1Lepton

    export PATH="$ANALYSIS_SOFTWARE/bin:$PATH"
    export PYTHONPATH="$ANALYSIS_BASE:$ANALYSIS_SOFTWARE/lib/python${vpython}/site-packages:$PYTHONPATH"


    #
    # helpers
    #

    #
    # setup law
    #

    export LAW_HOME="$ANALYSIS_BASE/tmp/.law"
    export LAW_CONFIG_FILE="$ANALYSIS_BASE/law.cfg"

    source "$( law completion )"
    law index --verbose
}
action "$@"
