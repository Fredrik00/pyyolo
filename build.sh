make -f Makefile_CPU
rm -rf build
python3 setup.py build
sudo python3 setup.py install
