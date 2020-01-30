DATA_DIR_BASE := ./data
UD_DIR_BASE := $(DATA_DIR_BASE)/ud

UDURL := https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3105/ud-treebanks-v2.5.tgz

UD_DIR := $(UD_DIR_BASE)/ud-treebanks-v2.5
UD_FILE := $(UD_DIR_BASE)/ud-treebanks-v2.5.tgz


get_ud: $(UD_DIR)
	echo "Finished getting UD data"

# Get Universal Dependencies data
$(UD_DIR):
	echo "Get ud data"
	mkdir -p $(UD_DIR_BASE)
	wget -P $(UD_DIR_BASE) $(UDURL)
	tar -xvzf $(UD_FILE) -C $(UD_DIR_BASE)
