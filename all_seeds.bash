#!/bin/bash

# Code for running multiple seeds and configurations in series

# e.g. bash all_seeds.bash sawyer_drawer_open mar23 0 true false

env=$1
exp_name=$2
gpu_i=$3
use_sqil=$4
nstep_off=$5
seeds=(1 2 3 4 5)

echo "Starting for loop of execution with args $@"
for seed in "${seeds[@]}"; do
    echo "Running seed ${seed}, env ${env}"
    date_str=$(date '+%m-%d-%y_%H_%M_%S')

    case ${env} in
        "relocate-human-v0")
            num_steps=1500000
            ;;
        "sawyer_drawer_open" | "sawyer_bin_picking" | "door-human-v0" | "sawyer_drawer_close")
            num_steps=300000
            ;;
        "sawyer_push" | "sawyer_lift" | "sawyer_box_close" | "hammer-human-v0")
            num_steps=500000
            ;;
        *)
            echo "unknown env! exiting"
            exit 1
            ;;
    esac

    python_args=(
        --gin_bindings="train_eval.env_name=\"${env}\""
        --gin_bindings="train_eval.num_iterations=${num_steps}"
        --gin_bindings="train_eval.gpu_i=${gpu_i}"
    )

    if [ ${env} = "sawyer_bin_picking" ]; then
        python_args+=(--gin_bindings='critic_loss.q_combinator="max"')
        python_args+=(--gin_bindings='actor_loss.q_combinator="max"')
    fi

    if [ "${use_sqil}" = "true" ] && [ "${nstep_off}" = "true" ]; then
        echo "Running with SQIL loss, nstep off"
        python_args+=(--gin_bindings='train_eval.n_step=None')
        python_args+=(--gin_bindings='critic_loss.loss_name="q"')
        python_args+=(--root_dir="results/$VPACE_TOP_DIR/${env}/${seed}/sqil_theirs_nstepoff/${exp_name}/${date_str}")
    elif [ "${use_sqil}" = "true" ]; then
        echo "Running with SQIL loss, nstep on"
        python_args+=(--gin_bindings='critic_loss.loss_name="q"')
        python_args+=(--root_dir="results/$VPACE_TOP_DIR/${env}/${seed}/sqil_theirs/${exp_name}/${date_str}")
    else
        python_args+=(--root_dir="results/$VPACE_TOP_DIR/${env}/${seed}/rce_theirs/${exp_name}/${date_str}")
        echo "Running with RCE loss/params."
    fi

    echo "All args: ${python_args[@]}"

    python train_eval.py "${python_args[@]}"
done