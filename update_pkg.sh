pip uninstall -y diff-gaussian-rasterization
rm -rf submodules/diff-gaussian-rasterization/build
rm -rf submodules/diff-gaussian-rasterization/diff_gaussian_rasterization.egg-info
pip install -e submodules/diff-gaussian-rasterization