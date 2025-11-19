#!/bin/bash
# Installation script for Heterodox Macro Dashboard

echo "============================================"
echo "Heterodox Macro Dashboard - Installation"
echo "============================================"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Install requirements
echo ""
echo "Installing Python dependencies..."
pip3 install -r ../requirements.txt

echo ""
echo "============================================"
echo "Installation complete!"
echo "============================================"
echo ""
echo "To run the application:"
echo "  python3 main.py"
echo ""
echo "or:"
echo "  chmod +x main.py"
echo "  ./main.py"
echo ""
echo "See README.md for full documentation."
echo ""
