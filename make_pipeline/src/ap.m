addpath(genpath('svm'));

traininputs = h5read(trainfile, '/traininputs');
trainoutputs = h5read(trainfile, '/trainoutputs');
testinputs = h5read(testfile, '/testinputs');
testoutputs = h5read(testfile, '/testoutputs');
ninit = 3

mean_aps = zeros(ninit, 1);
mean_accs = zeros(ninit, 1);

for initidx=1:ninit
    clear ap;

    fprintf('%d/%d) start classfication with chi-squared kernel svm, computing kernel matrices......\n', initidx, ninit);

    Xtrain = double(traininputs(:,:,initidx)');
    Xtest = double(testinputs(:,:,initidx)');
    train_labels = trainoutputs';
    test_labels = testoutputs';

    [Ktrain, Ktest] = compute_kernel_matrices(Xtrain, Xtest);

    counter = 0;
    average_ap = 0;
    average_acc = 0;

    for label=1:12
        Ytest_new = double(test_labels(1:end,label));
        Ytrain_new = double(train_labels(1:end,label));
        
        n_total = length(Ytrain_new);
        n_pos = sum(Ytrain_new);
        n_neg = n_total-n_pos;
        
        cost = 100; % chosen as fixed: on personal communication with Wang et. al [42] paper citation
        
        % positive and negative weights balanced with number of positive and
        % negative examples
        % on personal communication with Wang et. al [42] paper citation
        w_pos = n_total/(2*n_pos);
        w_neg = n_total/(2*n_neg);
        
        option_string = sprintf('-t 4 -q -s 0 -b 1 -c %f -w1 %f -w0 %f',cost, w_pos, w_neg);

        model = svmtrain(Ytrain_new, Ktrain, option_string);
        
        [~, accuracy,prob_estimates] = svmpredict(Ytest_new, Ktest, model, '-b 1');
        
        [ap,~,~] = computeAP(prob_estimates(:,model.Label==1), Ytest_new, 1, 0, 0);
            
        average_ap = average_ap + ap;
        average_acc = average_acc + accuracy(1);
        counter = counter + 1;
        
        fprintf('label = %d,ap = %f, w_neg = %f, w_pos = %f\n', label,  ap, w_neg, w_pos);    
    end
    mean_aps(initidx) = average_ap / 12;
    mean_accs(initidx) = average_acc / 12;
    fprintf('mean_ap = %f, mean_acc = %f\n', mean_aps(initidx), mean_accs(initidx));

end
fid = fopen(reportfile, 'w');
for initidx=1:ninit
    fprintf(fid, '%d) mean_ap = %f, mean_acc = %f\n', initidx, mean_aps(initidx), mean_accs(initidx));
end
fclose(fid);
