# Notes

ObjNav agent submitted to EvalAI for Habitat Challenge 2021. Accompanying "Auxiliary Tasks and Exploration Enable ObjectNav."
This code may be useful as a reference for a minimal implementation of the agent in the paper.

Main ObjNav repo: https://github.com/joel99/objectnav/

## Some notes on running this code
- Dockerfile is the ultimate source of truth. 2021 file is `objnav_aux.Dockerfile`
- Dockerfile will prepare source files and call the runscript. Run script is `submit_aux.sh`

1. Build the agent
docker build . --file objnav_aux.Dockerfile -t objectnav_submission

Inside the Dockerfile, check/update:
- the rednet ckpt (21 or 40 cat) - also update line 198 in `aux_agent` depending on which one you use
- agent ckpt
- agent config

2. Submit
./test_locally_objectnav_rgbd.sh --docker-name objectnav_submission
evalai push objectnav_submission:latest --phase habitat20-objectnav-minival
evalai push objectnav_submission:latest --phase habitat20-objectnav-test-std
<!-- add a --private or --public flag to the push to control submission visibility  -->
