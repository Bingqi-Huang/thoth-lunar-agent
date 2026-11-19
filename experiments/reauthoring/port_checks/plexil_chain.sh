#!/usr/bin/env bash
# PLEXIL plan through the real adapter against the stand-in boundary, on an isolated domain:
#   plexil_chain.sh <repo> <run_dir> <model_dir: .../plexil> <plan> <sim_boundary.py> [boundary args...]
# MISSION_GOAL / BASELINE_REVISION pass through.
set -eo pipefail
REPO=$1; RUN=$2; MODEL=$3; PLAN=$4; SB=$5; shift 5
mkdir -p $RUN
export ROS_DOMAIN_ID=93 PLEXIL_HOME=$HOME/Programs/plexil
export PATH=$PLEXIL_HOME/scripts:/usr/bin:/bin
source /opt/ros/humble/setup.bash
source $REPO/openral/install/setup.bash
source $REPO/ros/install/local_setup.bash
export LD_LIBRARY_PATH=$REPO/ros/install/lib:$PLEXIL_HOME/lib:$LD_LIBRARY_PATH
python3 $SB "$@" > $RUN/boundary.log 2>&1 &
B=$!
trap 'kill $B 2>/dev/null' EXIT
sleep 2
cd $REPO/baselines/plexil
timeout 600 $PLEXIL_HOME/bin/universalExec -c $MODEL/interface-config.xml -L $MODEL/plans/lib -p $MODEL/plans/$PLAN.plx > $RUN/exec.log 2>&1 || echo "exec rc=$?" >> $RUN/exec.log
sed -n 's/.*\[europa_sim_boundary\]: \(.* -> success=[A-Za-z]*\)$/\1/p' $RUN/boundary.log > $RUN/calls.txt
echo "$(wc -l < $RUN/calls.txt) calls; $(grep "^$PLAN " $RUN/exec.log | sed 's/  */ /g' | cut -c1-260)"
