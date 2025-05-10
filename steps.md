# ssh
`
ssh-keygen -t ed25519 -C "namkha1032@gmail.com"
`
`
eval "$(ssh-agent -s)"
`
`
ssh-add ~/.ssh/id_ed25519
`
`
cat ~/.ssh/id_ed25519.pub
cat ~/.ssh/id_rsa.pub
`

# git
`
git config --global user.name "namkha1032"
`
`
git config --global user.email "namkha.nguyen@student.adelaide.edu.au"
`


# transfer
```
scp -r /home/namkha/Documents/mydev/transformer-zero/working cloud-gpu:/workspace/
```
```
scp -r cloud-gpu:/workspace/working/ /home/namkha/Documents/mydev/transformer-zero/
```

# miniconda
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
```
bash Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda3
```
```
source ~/miniconda3/bin/activate
```
# conda env
```
conda create -n nk python=3.10
```
```
conda activate nk
```
```
pip install -r requirements.txt
```
# gdown
```
cd ../
```
```
gdown 1lmyxFFVyxejTIxO-VNcNBFZJLwT8YzMw
gdown 1MxHArU50zM6dZk1AymLv2UgDebC2yDFI
```
```
unzip flowers.zip
```
```
tensorboard --logdir logs --host 0.0.0.0 --port 6007
```
