cd ~/
#sh -c 'mv utils_thisbuild content'
cd ~/utils_thisbuild

# download necessary model file and map file 
sudo /home/docker/.local/bin/gdown --id "11dHZ_Fl2DLkfxlg0_4ayimRgjowLOkgs"
sudo /home/docker/.local/bin/gdown --id "1uFBbk0iqyCDAAnrs8jqNgz3T3I9ewuB5"
sudo chown -R ~/utils_thisbuild
tar -xvf presets.tar.gz
#sudo /home/docker/.local/bin/gdown --id "1pvAkaCxt9UIHt6SE-MJtg0ckGjUFZ569"
#sudo /home/docker/.local/bin/gdown --id "1Qcg_LkW0nRO5932OP29nuR_B1yMi6VVA"
#sudo /home/docker/.local/bin/gdown --id "1PNp2a7dlXH8AiovEUV5EpjtZdSnVz5v6"
#sh -c 'mkdir -p ~/cei_mandarin_tts/content'
#cd ~/cei_mandarin_tts/content
#sh -c 'ln -s ~/utils_thisbuild/model-100000.h5 tacotron2-100k.h5'
#sh -c 'ln -s ~/utils_thisbuild/model-200000.h5 fastspeech2-200k.h5'
#sh -c 'ln -s ~/utils_thisbuild/generator-920000.h5 mb.melgan-920k.h5'
#sh -c 'ln -s ~/utils_thisbuild/baker_mapper.json baker_mapper.json'
# merge all id_rsa.pub into this container
#sh -c 'cat /home/docker/utils_thisbuild/*pub > /home/docker/.ssh/authorized_keys'
sh -c 'git config --global user.name alvinyc'
sh -c 'git config --global user.email chen.yongcheng@gmail.com'


rsync -av --exclude='*sh' ~/utils_thisbuild ~/cei_pytorch_vc/ 
#basically, this folder need the following file
# 1.[file] 20171222_deepvoice3_vctk108_checkpoint_step000300000.pth
# 2.[folder] presets
#        - [file] deepvoice3_ljspeech.json
#        - [file] deepvoice3_niklm.json
#        - [file] deepvoice3_nikls.json
#        - [file] deepvoice3_vctk.json
#        - [file] nyanko_ljspeech.json
python -c "import nltk; nltk.download('cmudict')"


