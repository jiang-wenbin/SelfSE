# PYTHONPATH
export MAIN_ROOT=$PWD/../../
export LC_ALL=C
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=$MAIN_ROOT:$PYTHONPATH
if [ -d "$HOME/workspace/kaldi" ]; then
    export KALDI_ROOT="$HOME/workspace/kaldi"
elif [ -d "$HOME/shared/workspace/kaldi" ]; then
    export KALDI_ROOT="$HOME/shared/workspace/kaldi"
fi