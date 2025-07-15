echo "Cleaning previous build artifacts..."
rm -rf build/
rm -rf _skbuild/
rm -rf ataraxai_assistant.egg-info/
rm -rf dist/
rm -f ataraxai/hegemonikon_py*.so 


pip uninstall ataraxai_assistant -y || echo "Package not previously installed or uninstall failed."
echo "Cleaning complete."