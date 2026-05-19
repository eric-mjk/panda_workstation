# Docker commit cleanup

conda clean --all -y
apt-get clean && rm -rf /var/lib/apt/lists/*
rm -rf /tmp/*
find /var/log -type f -delete
