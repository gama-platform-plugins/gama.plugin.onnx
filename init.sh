#!/bin/bash
# Initialize this repo from the plugin template.
#
# Usage:
#   bash init.sh <plugin-id> <plugin-label>
#
# Arguments:
#   plugin-id     OSGi bundle ID with exactly 3 dot-separated parts, e.g. gama.plugin.flooding
#   plugin-label  Human-readable name, e.g. "Flooding Simulation"
#
# The feature ID is automatically derived by inserting .feature. before the third part:
#   org.example.myplugin → org.example.feature.myplugin

set -e

PLUGIN_ID="$1"
PLUGIN_LABEL="$2"

if [[ -z "$PLUGIN_ID" || -z "$PLUGIN_LABEL" ]]; then
    echo "Usage: bash init.sh <plugin-id> <plugin-label>"
    echo "  e.g. bash init.sh gama.plugin.flooding \"Flooding Simulation\""
    exit 1
fi

if [[ ! "$PLUGIN_ID" =~ ^[^.]+\.[^.]+\.[^.]+$ ]]; then
    echo "ERROR: plugin-id must have exactly 3 dot-separated parts (e.g. gama.plugin.flooding)"
    exit 1
fi

if [[ ! -d MY_PLUGIN ]]; then
    echo "ERROR: MY_PLUGIN/ not found — has this repo already been initialized?"
    exit 1
fi

# Derive feature ID: insert .feature. before the third part
PART1="${PLUGIN_ID%%.*}"
REST="${PLUGIN_ID#*.}"
PART2="${REST%%.*}"
PART3="${REST#*.}"
FEATURE_ID="${PART1}.${PART2}.feature.${PART3}"

echo "Plugin ID : $PLUGIN_ID"
echo "Feature ID: $FEATURE_ID"
echo "Label     : $PLUGIN_LABEL"
echo ""

# Derive short name and Java class name from plugin ID
PLUGIN_SHORT="${PLUGIN_ID##*.}"   # gama.plugin.flooding → flooding
CLASS_NAME="$(tr '[:lower:]' '[:upper:]' <<< "${PLUGIN_SHORT:0:1}")${PLUGIN_SHORT:1}Skill"
PACKAGE_PATH="${PLUGIN_ID//.//}"  # gama.plugin.flooding → gama/plugin/flooding

# ── 1. Rename directories ────────────────────────────────────────────────────
echo "Renaming directories..."
mv MY_PLUGIN "$PLUGIN_ID"
mv MY_PLUGIN.feature "$FEATURE_ID"

# ── 2. Plugin bundle (MANIFEST.MF, pom.xml) ─────────────────────────────────
echo "Updating plugin files..."
sed -i \
    -e "s/Bundle-Name: MY_PLUGIN/Bundle-Name: $PLUGIN_LABEL/" \
    -e "s/Bundle-SymbolicName: gama\.plugin\.MY_PLUGIN/Bundle-SymbolicName: $PLUGIN_ID/" \
    -e "s/Automatic-Module-Name: gama\.plugin\.MY_PLUGIN/Automatic-Module-Name: $PLUGIN_ID/" \
    "$PLUGIN_ID/META-INF/MANIFEST.MF"

sed -i "s/gama\.plugin\.MY_PLUGIN/$PLUGIN_ID/g" "$PLUGIN_ID/pom.xml"

sed -i "s/MY_PLUGIN/$PLUGIN_ID/g" "$PLUGIN_ID/.project"

# ── 3. Rename Java skill (package dir + class name) ─────────────────────────
echo "Renaming Java skill..."
mkdir -p "$PLUGIN_ID/src/$PACKAGE_PATH"
mv "$PLUGIN_ID/src/gama/plugin/MY_PLUGIN/MySkill.java" \
   "$PLUGIN_ID/src/$PACKAGE_PATH/${CLASS_NAME}.java"
rm -rf "$PLUGIN_ID/src/gama/plugin/MY_PLUGIN"

sed -i \
    -e "s/package gama\.plugin\.MY_PLUGIN/package ${PLUGIN_ID}/" \
    -e "s/my_skill/${PLUGIN_SHORT}_skill/" \
    -e "s/my_action/${PLUGIN_SHORT}_action/" \
    -e "s/MySkill/${CLASS_NAME}/g" \
    "$PLUGIN_ID/src/$PACKAGE_PATH/${CLASS_NAME}.java"

# ── 4. Feature (feature.xml, pom.xml) ───────────────────────────────────────
echo "Updating feature files..."
sed -i \
    -e "s/id=\"gama\.plugin\.feature\.MY_PLUGIN\"/id=\"$FEATURE_ID\"/" \
    -e "s/label=\"MY_PLUGIN\"/label=\"$PLUGIN_LABEL\"/" \
    -e "s/id=\"gama\.plugin\.MY_PLUGIN\"/id=\"$PLUGIN_ID\"/" \
    "$FEATURE_ID/feature.xml"

sed -i "s/gama\.plugin\.feature\.MY_PLUGIN/$FEATURE_ID/g" "$FEATURE_ID/pom.xml"

sed -i "s/MY_PLUGIN.feature/$FEATURE_ID/g" "$FEATURE_ID/.project"

# ── 5. Parent POM modules ────────────────────────────────────────────────────
echo "Updating parent/pom.xml modules..."
sed -i \
    -e "s|../MY_PLUGIN.feature|../$FEATURE_ID|" \
    -e "s|../MY_PLUGIN|../$PLUGIN_ID|" \
    gama.plugin.parent/pom.xml

# ── 6. p2updatesite category.xml ────────────────────────────────────────────
echo "Updating p2updatesite/category.xml..."
sed -i \
    -e "s/gama\.plugin\.feature\.MY_PLUGIN/$FEATURE_ID/g" \
    -e "s/MY_PLUGIN/$PLUGIN_LABEL/g" \
    gama.plugin.p2updatesite/category.xml

# ── 7. Remove template boilerplate ──────────────────────────────────────────
echo "Removing template README..."
rm README.md

echo ""
echo "Done! Next steps:"
echo "  1. Pull this commit locally"
echo "  2. Open this project in Eclipse (both the plugin and the feature)"
echo "  3. Add your plugin code under $PLUGIN_ID/src/"
echo "  4. Declare OSGi dependencies in $PLUGIN_ID/META-INF/MANIFEST.MF (Require-Bundle)"
echo "  5. (Optional) Adjust the category in p2updatesite/category.xml"
echo "  6. Add your own README.md"
echo "  7. Push to trigger the build"
