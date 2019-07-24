# TODO: add check if file is ok, else delete it (e.g. if sample file is
# 		not ok, delete it, so it is built again)
# TODO: only gensamples-trainmodel finished so far, other settings still missing


include config.mk


#######TARGETS#######

# default target
all: $(REPORTFILE)

# trick to make sure necessary directories exist (depend on <dirname>/.d)

$(RESULTSDIR)/.d: 
	mkdir -p $(@D)
	touch $@

$(TMP_DIR)/.d:
	mkdir -p $(@D)
	touch $@
	touch $(TMP_DIR)/.excludeme
	

# preprocessing step
preprocessing: $(SAMPLES_WHITE_FILE)

# generate samples
$(SAMPLES_FILE): $(RESULTSDIR)/.d $(TMP_DIR)/.d $(DATADIR) \
			 	 $(CFGFILE) $(SRCDIR)/gensamples.py
	$(PYTHON) $(SRCDIR)/gensamples.py $(CFGPARAM) --videodir=$(VIDEODIR) \
		--clipsetdir=$(CLIPSETDIR) --labelsfile=$(LABELSFILE) --samplesfile=$@

# compute pca params
$(PCA_FILE): $(SAMPLES_FILE) $(SRCDIR)/compute_pca_params.py
	$(PYTHON) $(SRCDIR)/compute_pca_params.py $(CFGPARAM) \
		--samplesfile=$(SAMPLES_FILE) --pcafile=$@

# whiten data
$(SAMPLES_WHITE_FILE): $(PCA_FILE) $(SRCDIR)/whiten_data.py
	$(PYTHON) $(SRCDIR)/whiten_data.py $(CFGPARAM) \
		--samplesfile=$(SAMPLES_FILE) --pcafile=$(PCA_FILE) \
		--sampleswhitefile=$@

# FIXME: graddescent.load_params() should support hdf5!!
# train the model
$(MODELFILE): $(SAMPLES_WHITE_FILE) $(SRCDIR)/trainmodel.py
	$(PYTHON) $(SRCDIR)/trainmodel.py $(CFGPARAM) \
		--sampleswhitefile=$(SAMPLES_WHITE_FILE) \
		--trainerfile=$(TRAINERFILE) \
		--tmpdir=$(TMP_DIR) --modelfile=$@

$(KMEANSTRAINFILE): $(MODELFILE)
	$(PYTHON) $(SRCDIR)/gen_clustering_traindata.py $(CFGPARAM) \
		--videodir=$(VIDEODIR) --clipsetdir=$(CLIPSETDIR) \
		--labelsfile=$(LABELSFILE) \
		--sampleswhitefile=$(SAMPLES_WHITE_FILE) \
		--modelfile=$(MODELFILE) \
		--pcafile=$(PCA_FILE) \
		--nthreads=$(NTHREADS) \
		--kmeanstrainfile=$@

# get cluster centers using KMEANS
$(KMEANSFILE): $(KMEANSTRAINFILE)
	$(PYTHON) $(SRCDIR)/get_centroids.py $(CFGPARAM) \
		--kmeanstrainfile=$(KMEANSTRAINFILE) --kmeansfile=$@

# FIXME: some bug with h5read & h5insert error msg?? ... also split this into 
# 		 two calls (for train and test data, so we can use -j2 flag (for train
# 		 &test)
# TODO: add nthreads option to config and read it in gen_classifierdata.py

# FIXME: how to make two dependencies without running two targets (or one twice)
# 		 and without abort "no rule for target" (CLASSIFIER_TRAINFILE)
# generate classifier data
$(CLASSIFIER_TESTFILE): $(SRCDIR)/gen_classifierdata.py $(KMEANSFILE)
	$(PYTHON) $(SRCDIR)/gen_classifierdata.py $(CFGPARAM) \
		--sampleswhitefile=$(SAMPLES_WHITE_FILE) \
		--modelfile=$(MODELFILE) --kmeansfile=$(KMEANSFILE) \
		--classifiertrainfile=$(CLASSIFIER_TRAINFILE) \
		--classifiertestfile=$(CLASSIFIER_TESTFILE) \
		--videodir=$(VIDEODIR) --clipsetdir=$(CLIPSETDIR) \
		--labelsfile=$(LABELSFILE) \
		--kmeanstrainfile=$(KMEANSTRAINFILE)

$(REPORTFILE): $(SRCDIR)/ap.m $(CLASSIFIER_TESTFILE)
	matlab -nojvm -nodesktop -r "trainfile='$(CLASSIFIER_TRAINFILE)'; \
								 testfile='$(CLASSIFIER_TESTFILE)'; \
								 reportfile='$@'; \
								 run '$(SRCDIR)/ap'; \
								 quit"

gensamples: $(SAMPLES_FILE)


skippreprocessing:
	touch tmp/samples.h5
	touch pca_params.h5
	touch tmp/samples_white.h5

clean:
	-find $(SRCDIR)/ -name "*.pyc" -exec rm -rf {} \;
	-rm $(MODELFILE)
	-rm $(MODELFILE)_bak
	-rm $(TRAINERFILE)
	-rm $(TRAINERFILE)_bak
	-rm -rf tmp
	-rm .d
	-rm $(SAMPLES_WHITE_FILE)
	-rm $(KMEANSFILE)
	-rm $(CLASSIFIER_TESTFILE)
	-rm $(CLASSIFIER_TRAINFILE)

