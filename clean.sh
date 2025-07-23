echo "Cleaning previous build artifacts..."
rm -rf build/
rm -rf _skbuild/
rm -rf ataraxai.egg-info/
rm -rf dist/
rm -f ataraxai/hegemonikon_py*.so 


pip uninstall ataraxai -y || echo "Package not previously installed or uninstall failed."
echo "Cleaning complete."