# Setup rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# Update the os
sudo apt-get update
sudo apt-get install libwebkit2gtk-4.1-dev build-essential curl wget libssl-dev libgtk-3-dev libayatana-appindicator3-dev librsvg2-dev
# Frontend dependencies
cd ataraxai-ui
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
nvm install --lts
node -v
rm -rf node_modules
rm package-lock.json
npm install
npm run tauri dev
npm add -D @tauri-apps/cli
npm create tauri-app@latest ataraxai-ui
npm install lucide-react
npm run tauri dev