
Installation:
cd /.../stPack 
python3 setup.py install
cd speechTools
rm -R diverg diverg.cython*
cp ../build/lib*/speechTools/diverg ./ -r
cp ../build/lib*/speechTools/diverg.cython*
 

Extract and save features:
Place .wav files on directory data/wavs
python3 saveFeatures.py



