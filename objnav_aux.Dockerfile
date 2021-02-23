# * 2021
# TODO check if there are any newer dockerfiles with newer habitat-labs.
FROM fairembodied/habitat-challenge:testing_2020_habitat_base_docker

# RUN /bin/bash -c ". activate habitat; pip install torch torchvision; pip install ifcfg tensorboard"
RUN /bin/bash -c ". activate habitat; conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch; conda install tensorboard; pip install ifcfg"

# RUN /bin/bash -c ". activate habitat; pip install torch==1.7.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html; pip install ifcfg tensorboard"

RUN /bin/bash -c "git clone http://github.com/facebookresearch/habitat-api.git habitat-api2 && (cd habitat-api2 && git checkout 6cf86df56b7e9db118a1225354dd251579fe8d0c) && cp -r habitat-api2/habitat_baselines habitat-api/."

RUN pwd
ADD submit_aux.sh submit_aux.sh

ENV AGENT_EVALUATION_TYPE remote
ADD ckpts/rednet.pth ckpts/rednet.pth

ADD configs/challenge_objectnav2020.local.rgbd.yaml /challenge_objectnav2020.local.rgbd.yaml

# ADD ckpts/base-full.35.pth ckpts/aux.ckpt.pth
ADD ckpts/base4-full.35.pth ckpts/aux.ckpt.pth
# ADD ckpts/pt_sparse-full.35.pth ckpts/aux.ckpt.pth
# ADD ckpts/split_clamp-full.35.pth ckpts/aux.ckpt.pth

ADD configs/obj_base.yaml configs/obj_base.yaml
# ADD configs/base-full.yaml configs/aux_base.yaml
ADD configs/base4-full.yaml configs/aux_base.yaml
# ADD configs/pt_sparse-full.yaml configs/aux_base.yaml
# ADD configs/split-full.yaml configs/aux_base.yaml
ADD aux_agent.py aux_agent.py
ADD src/ src/


ENV TRACK_CONFIG_FILE "/challenge_objectnav2020.local.rgbd.yaml"
ENV AGENT_CONFIG_FILE "configs/aux_base.yaml"

CMD ["/bin/bash", "-c", "source activate habitat && export PYTHONPATH=/evalai-remote-evaluation:$PYTHONPATH && export CHALLENGE_CONFIG_FILE=$TRACK_CONFIG_FILE && bash submit_aux.sh --model-path ckpts/aux.ckpt.pth --config-path $AGENT_CONFIG_FILE"]
