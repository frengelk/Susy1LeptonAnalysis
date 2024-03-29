#!/usr/bin/env bash

# example for setup
action() {
    # delete conda env
    eval "$($CONDA_EXE shell.$(basename $SHELL) hook)"
    conda deactivate

    local origin="$( pwd )"
    local scram_cores="$(grep -c ^processor /proc/cpuinfo)"
    #[ -z "$scram_cores" ] && scram_cores="4"

    export SCRAM_ARCH="slc7_amd64_gcc700"
    # "$SCRAM_ARCH"
    export CMSSW_VERSION="CMSSW_10_3_0"
    # "$CMSSW_VERSION" CMSSW_10_2_13
    #export CMSSW_BASE="$CMSSW_BASE"
	export CMSSW_BASE="/nfs/dust/cms/user/$USER/Code/cmssw/CMSSW_10_3_0"



    if [ ! -f "$CMSSW_BASE/good" ]; then
        if [ -d "$CMSSW_BASE" ]; then
            echo "remove already installed software in $CMSSW_BASE"
            rm -rf "$CMSSW_BASE"
        fi

        echo "setting up $CMSSW_VERSION with $SCRAM_ARCH in $CMSSW_BASE"

        #/cvmfs/cms.cern.ch/slc7_amd64_gcc700/cms/cmssw/CMSSW_10_2_3

        source "/cvmfs/cms.cern.ch/cmsset_default.sh"
        mkdir -p "$( dirname "$CMSSW_BASE" )"
        cd "$( dirname "$CMSSW_BASE" )"
        scramv1 project CMSSW "$CMSSW_VERSION" || return "$?"
        cd "$CMSSW_VERSION/src"
        eval `scramv1 runtime -sh` || return "$?"
		cmsenv
		git clone https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
        # IMPORTANT: Checkout the recommended tag on the link above
        git clone https://github.com/cms-analysis/CombineHarvester.git CombineHarvester
		cd CombineHarvester/CombineTools/
		mkdir python3
		2to3 --output-dir=python3 -W -n python
        #2to3 --output-dir=python3 -W -n python
        scram b -j "$scram_cores"|| return "$?"


        #
        # custom topics
        #

        # git cms-init
        # git cms-merge-topic cms-egamma:EgammaPostRecoTools

        # Install CombinedLimit and CombineHarvester
        # git clone https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
        # cd HiggsAnalysis/CombinedLimit
        # git fetch origin
        # git checkout v8.0.1
        # scram b clean; scram b -j "$scram_cores"
        # cd "$CMSSW_BASE/src"
        # git clone https://github.com/cms-analysis/CombineHarvester.git CombineHarvester
        # scram b -j "$scram_cores"


        #
        # compile
        #

        #scram b -j "$scram_cores" || return "$?"

        cd "$origin"

        touch "$CMSSW_BASE/good"

  else

      source "/cvmfs/cms.cern.ch/cmsset_default.sh" ""
      cd "$CMSSW_BASE/src" || return "$?"
      eval `scramv1 runtime -sh` || return "$?"
      cd "$origin"


  fi

  . /cvmfs/grid.cern.ch/umd-c7ui-latest/etc/profile.d/setup-c7-ui-example.sh

  # add missing python packages to cmssw
  export EXTERNAL_CMSSW="/nfs/dust/cms/user/$USER/Code/cmssw_software/external_cmssw"

  export CONDA_PYTHONPATH="/nfs/dust/cms/user/$USER/Anaconda/bin/python"

  _addpy() {
  [ ! -z "$1" ] && export PYTHONPATH="$1:$PYTHONPATH"
  }
  #"$1:$PYTHONPATH"
  #"$CONDA_PYTHONPATH"

  _addbin() {
      [ ! -z "$1" ] && export PATH="$1:$PATH"
  }

  if [ ! -f "$EXTERNAL_CMSSW/.good" ]; then
      cmssw_install_pip() {
          pip install --ignore-installed --no-cache-dir --prefix "$EXTERNAL_CMSSW" "$@"
      }
      cmssw_install_pip order
      cmssw_install_pip scinum
      cmssw_install_pip luigi
      cmssw_install_pip tabulate
      cmssw_install_pip tqdm
	  cmssw_install_pip law
      LAW_INSTALL_EXECUTABLE=env cmssw_install_pip law --no-binary law
      ln -s /usr/lib/python2.7/dist-packages/gfal2.so $EXTERNAL_CMSSW/lib/python2.7/site-packages/
      touch "$EXTERNAL_CMSSW/.good"
  fi

  # add external python packages
  _addbin "$EXTERNAL_CMSSW/bin"
  _addpy "$EXTERNAL_CMSSW/lib/python2.7/site-packages"

  #export CRAB_SOURCE_SCRIPT="/cvmfs/cms.cern.ch/crab3/crab_slc7_standalone.sh"
  source /cvmfs/cms.cern.ch/crab3/crab.sh
}
action "$@"
