# Things to do

1. Dockerfile is the ultimate source of truth. 2021 file is `objnav_aux.Dockerfile`
2. Dockerfile will prepare source files and call the runscript. Run script is `submit_aux.sh`

## TODO
- [x] Update dockerfile
    - Pending up to date API.
- [x] Update config
    - Make sure we have parity at the task specification level as well
- [x] Update source
    """
        Source is largely a clone from `habitat-lab`, except
        1. Imports are changed from absolute to relative
        2. Hardcoded snippets are either copy-pasted in or removed from the code.
        3. Registry references are removed (registries are irrelevant)
    """
- [x] Update runner (agent.py)
- [ ] Test

# EvalAI commands
docker build . --file objnav_aux.Dockerfile -t objectnav_submission
./test_locally_objectnav_rgbd.sh --docker-name objectnav_submission
evalai push objectnav_submission:latest --phase habitat20-objectnav-minival
evalai push objectnav_submission:latest --phase habitat20-objectnav-test-std
<!-- add a --private or --public flag to the push to control submission visibility  -->

# Checklist - things I'm sure match
- belief_policy, policy
- encoder_dict
- obs_transformers
- default
- init
- resnet
- rednet
- running_mean_and_var
- rnn_state_encoder
- common
- coarsely, the evaluation loop
