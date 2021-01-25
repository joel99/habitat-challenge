FROM fairembodied/habitat-challenge:testing_2020_habitat_base_docker

RUN /bin/bash -c ". activate habitat; pip install torch==1.5.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html; pip install ifcfg tensorboard"

RUN /bin/bash -c "git clone http://github.com/facebookresearch/habitat-api.git habitat-api2 && (cd habitat-api2 && git checkout 6cf86df56b7e9db118a1225354dd251579fe8d0c) && cp -r habitat-api2/habitat_baselines habitat-api/."

RUN pwd
ADD aux_agent.py aux_agent.py
ADD submit_aux.sh submit_aux.sh
ADD configs/challenge_objectnav2020.local.rgbd.yaml /challenge_objectnav2020.local.rgbd.yaml

ADD configs/ configs/
ADD ckpts/nav-std.33.pth ckpts/aux.ckpt.pth

ADD src/ src/
ENV AGENT_EVALUATION_TYPE remote

ENV TRACK_CONFIG_FILE "/challenge_objectnav2020.local.rgbd.yaml"

CMD ["/bin/bash", "-c", "source activate habitat && export PYTHONPATH=/evalai-remote-evaluation:$PYTHONPATH && export CHALLENGE_CONFIG_FILE=$TRACK_CONFIG_FILE && bash submit_aux.sh --model-path ckpts/aux.ckpt.pth --config-path configs/nav.yaml"]
