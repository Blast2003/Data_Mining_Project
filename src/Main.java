package src;

import weka.core.converters.CSVLoader;
import weka.core.converters.ArffSaver;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.Standardize;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.AddCluster;
import weka.clusterers.SimpleKMeans;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.J48;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.AbstractClassifier;
import weka.gui.visualize.VisualizePanel;
import weka.gui.visualize.PlotData2D;
import weka.classifiers.evaluation.ThresholdCurve;

import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.List;
import java.util.stream.Collectors;

public class Main {

    // === Configurable parameters ===
    private static final String INPUT_CSV = "data/World Happiness Report 2024.csv";
    private static final String CLEANED_ARFF = "data/world_happiness_cleaned.arff";
    private static final String LATEST_PER_COUNTRY_ARFF = "data/world_happiness_latest_by_country.arff";
    private static final int CLUSTER_K = 6; // used by AddCluster and visualization
    private static final int RF_NUM_TREES = 100;
    private static final int CV_FOLDS = 10;
    private static final long RANDOM_SEED = 1L;
    private static final int REPEATED_CV_RUNS = 5; // number of repeated CV runs used in extended evaluation
    // Toggle: if want to disable GUI plots, set to false
    private static final boolean ENABLE_VIS = true;
    private static final int SAMPLE_PREDICTIONS = 10;
    // ================================

    public static void main(String[] args) {
        try {
            System.out.println("=== DATA MINING PIPELINE START ===");
            System.out.println("Loading CSV: " + INPUT_CSV);

            // 1. Load CSV
            Instances data = loadCSV(INPUT_CSV);
            if (data == null) {
                System.err.println("Failed to load CSV. Exiting.");
                return;
            }
            System.out.printf("Loaded: %d instances, %d attributes.%n", data.numInstances(), data.numAttributes());

            // ---------- STEP 1: Preprocessing ----------
            System.out.println("\n--- STEP 1: Preprocessing ---");
            data = removeEmptyAttributes(data);
            data = removeDuplicateInstances(data);

            // Winsorize numeric outliers (IQR)
            data = winsorizeNumericIQR(data);

            // Replace missing values with mean/mode
            data = replaceMissingValues(data);

            // b. Data analysis outputs (missingness, numeric summary, correlations, sample
            // table)
            System.out.println("\n--- STEP 1b: Data analysis (required) ---");
            printMissingSummary(data);
            printNumericSummary(data);
            printCorrelationMatrix(data);
            printCountsByYearAndTopCountries(data, 10);
            System.out.println("\nSample data (first 20 rows):");
            printSampleTable(data, 20);

            // Find precise life_ladder attribute (prefer exact 'life ladder' phrase)
            String lifeLadderName = findColumnNamePrecise(data, "life ladder", "life_ladder", "life ladder",
                    "life-ladder");
            if (lifeLadderName == null) {
                System.err.println("ERROR: could not find 'Life Ladder' column. Exiting Step 1.");
                return;
            } else {
                System.out.println("Detected life_ladder column: '" + lifeLadderName + "'");
            }

            // Add nominal target happiness_level and remove the numeric life_ladder =>
            // Create classification target happiness_level, 3 classes (Low/Medium/High) by
            // 33% / 66% quantiles
            data = addHappinessLevelFromLifeLadder(data, lifeLadderName);
            System.out.println(
                    "After preprocessing: " + data.numInstances() + " rows, " + data.numAttributes() + " attributes.");

            // Save cleaned ARFF / export cleaned dataset to ARFF for Weka/Java modelling
            saveArff(data, CLEANED_ARFF);
            System.out.println("Saved cleaned ARFF: " + CLEANED_ARFF);

            // Save aggregated / produce a shorter dataset (one row per country using the
            // latest year)
            // optional
            Instances latestPerCountry = buildLatestPerCountry(data);
            saveArff(latestPerCountry, LATEST_PER_COUNTRY_ARFF);
            System.out.println("Saved latest-per-country ARFF: " + LATEST_PER_COUNTRY_ARFF);

            // Ensure class index set to last attribute (happiness_level)
            data.setClassIndex(data.numAttributes() - 1);
            latestPerCountry.setClassIndex(latestPerCountry.numAttributes() - 1);

            // ---------- STEP 2: RandomForest (Algorithm A) ----------
            System.out.println("\n--- STEP 2: RandomForest (Algorithm A) ---");
            Instances standardizedForRF = standardizeIfNeeded(data); // standardized copy (class index preserved)

            RandomForest rf = new RandomForest();
            try {
                String[] rfOptions = {
                        "-I", Integer.toString(RF_NUM_TREES),
                        "-S", Integer.toString((int) RANDOM_SEED),
                        "-num-slots", "1"
                };
                rf.setOptions(rfOptions);
            } catch (Exception ex) {
                System.out
                        .println("Warning: could not set RandomForest options using setOptions(): " + ex.getMessage());
            }

            Evaluation evalRF = new Evaluation(standardizedForRF);
            long t0 = System.currentTimeMillis();
            evalRF.crossValidateModel(rf, standardizedForRF, CV_FOLDS, new Random(RANDOM_SEED));
            long t1 = System.currentTimeMillis();
            double rfTimeMs = (t1 - t0);
            System.out.println("RandomForest (10-fold CV) results:");
            System.out.println(evalRF.toSummaryString());
            System.out.println(evalRF.toClassDetailsString());
            System.out.println(evalRF.toMatrixString());
            System.out.printf("RandomForest CV time (ms): %.0f%n", rfTimeMs);

            // Train RF on full standardized dataset for demonstration (and ROC
            // visualization)
            try {
                rf.buildClassifier(standardizedForRF);
                // Print sample predictions for RandomForest (first N rows)
                printSamplePredictions(rf, standardizedForRF, SAMPLE_PREDICTIONS);
            } catch (Exception ex) {
                System.out.println(
                        "Warning: could not build RF on full dataset for sample predictions: " + ex.getMessage());
            }

            // ----------------------------
            // A — ROC curves for RandomForest (illustrative)
            // ----------------------------
            if (ENABLE_VIS) {
                try {
                    System.out.println(
                            "\n[Visual] Building RandomForest on full standardized data for ROC visualization...");
                    Evaluation evalForROC = new Evaluation(standardizedForRF);
                    evalForROC.evaluateModel(rf, standardizedForRF);

                    ThresholdCurve tc = new ThresholdCurve();
                    VisualizePanel vpROC = new VisualizePanel();
                    vpROC.setName("ROC curves - RandomForest (training set)");

                    for (int classIndex = 0; classIndex < standardizedForRF.numClasses(); classIndex++) {
                        Instances curve = tc.getCurve(evalForROC.predictions(), classIndex);
                        PlotData2D pd = new PlotData2D(curve);
                        pd.setPlotName("ROC class=" + standardizedForRF.classAttribute().value(classIndex));
                        pd.addInstanceNumberAttribute();
                        vpROC.addPlot(pd);
                    }

                    final VisualizePanel finalVpROC = vpROC;
                    EventQueue.invokeLater(() -> {
                        try {
                            JFrame jf = new JFrame("RandomForest ROC curves");
                            jf.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
                            jf.getContentPane().setLayout(new BorderLayout());
                            jf.getContentPane().add(finalVpROC, BorderLayout.CENTER);
                            jf.setSize(800, 600);
                            jf.setLocationRelativeTo(null);
                            jf.setVisible(true);
                        } catch (Exception ex) {
                            ex.printStackTrace();
                        }
                    });
                    System.out.println("[Visual] ROC window launched.");
                } catch (Exception ex) {
                    System.out.println("ROC visualization failed: " + ex.getMessage());
                }
            }

            // ---------- STEP 3: J48 + AddCluster (Algorithm B) ----------
            // Now we add cluster as a predictor, but KEEP 'happiness_level' as the class
            // target
            System.out.println("\n--- STEP 3: J48 (Algorithm B) with AddCluster (SimpleKMeans) ---");

            Instances dataWithCluster = addClusterAttribute(data, CLUSTER_K);

            // find happiness_level attribute and set as class (so J48 predicts same target
            // as RF)
            int hIdx = -1;
            for (int a = 0; a < dataWithCluster.numAttributes(); a++) {
                if (dataWithCluster.attribute(a).name().equalsIgnoreCase("happiness_level")) {
                    hIdx = a;
                    break;
                }
            }
            if (hIdx == -1) {
                // fallback: keep last as class but warn
                dataWithCluster.setClassIndex(dataWithCluster.numAttributes() - 1);
                System.out.println(
                        "Warning: 'happiness_level' not found in dataWithCluster. Class set to last attribute.");
            } else {
                dataWithCluster.setClassIndex(hIdx);
                System.out.println("Set class to 'happiness_level' (index " + (hIdx + 1) + ").");
            }

            // visualize clusters right after AddCluster so user inspects clusters before
            // training J48
            if (ENABLE_VIS) {
                visualizeClustersUsingClusterAttr(dataWithCluster, CLUSTER_K);
            }

            J48 j48 = new J48();
            j48.setConfidenceFactor(0.25f);
            j48.setMinNumObj(5);

            Evaluation evalJ48 = new Evaluation(dataWithCluster);
            long t2 = System.currentTimeMillis();
            evalJ48.crossValidateModel(j48, dataWithCluster, CV_FOLDS, new Random(RANDOM_SEED));
            long t3 = System.currentTimeMillis();
            double j48TimeMs = (t3 - t2);
            System.out.println("J48 (+ AddCluster And 10-fold CV) results:");
            System.out.println(evalJ48.toSummaryString());
            System.out.println(evalJ48.toClassDetailsString());
            System.out.println(evalJ48.toMatrixString());
            System.out.printf("J48 CV time (ms): %.0f%n", j48TimeMs);

            // Train final J48 on full dataWithCluster and print textual tree for
            // interpretability
            j48.buildClassifier(dataWithCluster);
            System.out.println("\n=== Final trained J48 tree (textual) ===");
            System.out.println(j48);

            // Print sample predictions for J48 (first N rows) — uses dataWithCluster
            // (cluster used as predictor)
            printSamplePredictions(j48, dataWithCluster, SAMPLE_PREDICTIONS);

            // ---------- NEW STEP: J48 (plain, no AddCluster) ----------
            // Run a plain J48 on the original preprocessed dataset (no cluster attribute)
            System.out.println("\n--- STEP 3b: J48 (plain) - no AddCluster (NEW) ---");
            J48 j48Plain = new J48();
            j48Plain.setConfidenceFactor(0.25f);
            j48Plain.setMinNumObj(5);

            // data already has class index set to last attribute (happiness_level)
            Evaluation evalJ48Plain = new Evaluation(data);
            long tPlainStart = System.currentTimeMillis();
            evalJ48Plain.crossValidateModel(j48Plain, data, CV_FOLDS, new Random(RANDOM_SEED));
            long tPlainEnd = System.currentTimeMillis();
            double j48PlainTimeMs = (tPlainEnd - tPlainStart);

            System.out.println("J48 (plain, 10-fold CV) results:");
            System.out.println(evalJ48Plain.toSummaryString());
            System.out.println(evalJ48Plain.toClassDetailsString());
            System.out.println(evalJ48Plain.toMatrixString());
            System.out.printf("J48 (plain) CV time (ms): %.0f%n", j48PlainTimeMs);

            // Train final plain J48 on full original data and print textual tree
            j48Plain.buildClassifier(data);
            System.out.println("\n=== Final trained J48 (plain) tree (textual) ===");
            System.out.println(j48Plain);

            // Print sample predictions for plain J48
            printSamplePredictions(j48Plain, data, SAMPLE_PREDICTIONS);

            // ---------- STEP 4: Comparison and detailed evaluation ----------
            // Keep the original RF vs J48+AddCluster comparison unchanged, and add a new
            // comparison between J48 plain and J48+AddCluster.
            System.out.println("\n--- STEP 4a: Comparison summary (RandomForest vs J48 + AddCluster) ---");
            printComparisonSummary(evalRF, rfTimeMs, evalJ48, j48TimeMs);

            // --- Repeated (paired) CV comparison between RF and J48+AddCluster (unchanged)
            System.out.println("\n--- STEP 4b: Repeated (paired) CV comparison and aggregated metrics (RandomForest vs J48+AddCluster) ---");
            RepeatedCVResult resRFvsJ48Cluster = repeatedPairedCrossValidation(
                    rf, standardizedForRF,
                    j48, dataWithCluster,
                    CV_FOLDS, REPEATED_CV_RUNS, RANDOM_SEED);
            resRFvsJ48Cluster.printSummary();

            // ---------- NEW: Comparison between J48 plain and J48 + AddCluster ----------
            System.out.println("\n--- STEP 4c (NEW): Comparison summary (J48 plain vs J48 + AddCluster) ---");
            System.out.println("Summary (Accuracy, Kappa, CV time ms):");
            System.out.printf(" J48 (plain)                : Accuracy=%.4f, Kappa=%.4f, Time(ms)=%.0f%n",
                    (1 - evalJ48Plain.errorRate()), evalJ48Plain.kappa(), j48PlainTimeMs);
            System.out.printf(" J48 (+ AddCluster KMeans)  : Accuracy=%.4f, Kappa=%.4f, Time(ms)=%.0f%n",
                    (1 - evalJ48.errorRate()), evalJ48.kappa(), j48TimeMs);
            System.out.println("Refer to the per-model details printed above for class-level metrics and confusion matrices.");

            System.out.println("\n--- STEP 4d (NEW): Repeated paired CV comparison (J48 plain vs J48 + AddCluster) ---");
            RepeatedCVResult resJ48Plain_vs_J48Cluster = repeatedPairedCrossValidation(
                    j48Plain, data,
                    j48, dataWithCluster,
                    CV_FOLDS, REPEATED_CV_RUNS, RANDOM_SEED);
            resJ48Plain_vs_J48Cluster.printSummary();

            System.out.println("\n=== PIPELINE FINISHED ===");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // --------------------------
    // === Repeated CV comparison helper and result class ===
    // --------------------------

    /**
     * Container for repeated CV aggregated results
     */
    public static class RepeatedCVResult {
        public int totalInstances = 0;
        public int correctA = 0;
        public int correctB = 0;
        public long totalTimeMsA = 0;
        public long totalTimeMsB = 0;
        public int[][] confA; // [actual][predicted]
        public int[][] confB;
        public List<Double> runAccuraciesA = new ArrayList<>();
        public List<Double> runAccuraciesB = new ArrayList<>();
        public long mcnemar_b = 0; // A correct, B wrong
        public long mcnemar_c = 0; // A wrong, B correct
        public String[] classNames;

        public RepeatedCVResult(int numClasses) {
            confA = new int[numClasses][numClasses];
            confB = new int[numClasses][numClasses];
        }

        private static double mean(List<Double> vals) {
            double s = 0;
            for (double v : vals)
                s += v;
            return s / vals.size();
        }

        private static double std(List<Double> vals) {
            double m = mean(vals);
            double s = 0;
            for (double v : vals)
                s += (v - m) * (v - m);
            return Math.sqrt(s / vals.size());
        }

        public void printSummary() {
            System.out.printf("Repeated CV runs: %d%n", runAccuraciesA.size());
            System.out.printf("Total test instances across runs: %d%n", totalInstances);
            System.out.println();

            System.out.printf("Model A per-run accuracies: %s%n", runAccuraciesA);
            System.out.printf("Model B per-run accuracies: %s%n", runAccuraciesB);
            System.out.println();

            System.out.printf("Model A mean accuracy = %.4f  (std = %.4f)%n", mean(runAccuraciesA),
                    std(runAccuraciesA));
            System.out.printf("Model B mean accuracy = %.4f  (std = %.4f)%n", mean(runAccuraciesB),
                    std(runAccuraciesB));
            System.out.println();

            System.out.printf("Aggregated accuracy A = %.4f%n", (double) correctA / totalInstances);
            System.out.printf("Aggregated accuracy B = %.4f%n", (double) correctB / totalInstances);

            double kappaA = computeKappa(confA, totalInstances);
            double kappaB = computeKappa(confB, totalInstances);
            System.out.printf("Aggregated Cohen's kappa A = %.4f%n", kappaA);
            System.out.printf("Aggregated Cohen's kappa B = %.4f%n", kappaB);
            System.out.println();

            System.out.println("Aggregated confusion matrix A (rows=true, cols=pred):");
            printConfMatrix(confA, classNames);
            System.out.println();
            System.out.println("Aggregated confusion matrix B (rows=true, cols=pred):");
            printConfMatrix(confB, classNames);
            System.out.println();

            System.out.println("Aggregated per-class Precision / Recall / F1 (Model A):");
            printPerClassPRF(confA, classNames);
            System.out.println("Aggregated per-class Precision / Recall / F1 (Model B):");
            printPerClassPRF(confB, classNames);
            System.out.println();

            // McNemar
            System.out.printf("McNemar counts: b (A correct, B wrong) = %d, c (A wrong, B correct) = %d%n", mcnemar_b,
                    mcnemar_c);
            if (mcnemar_b + mcnemar_c == 0) {
                System.out.println("McNemar: no discordant pairs (b+c == 0). Cannot test significance.");
            } else {
                // continuity correction
                double chi2 = ((Math.abs(mcnemar_b - mcnemar_c) - 1.0) * (Math.abs(mcnemar_b - mcnemar_c) - 1.0))
                        / (mcnemar_b + mcnemar_c);
                System.out.printf("McNemar chi-square (with continuity correction) = %.4f%n", chi2);
                System.out.println("Compare to chi-square(1 df) critical value 3.841 (alpha=0.05).");
                if (chi2 > 3.841)
                    System.out.println("=> Difference is statistically significant at alpha=0.05 (reject H0).");
                else
                    System.out.println("=> Not significant at alpha=0.05 (fail to reject H0).");
            }
        }
    }

    /**
     * Perform repeated (paired) stratified cross-validation for two classifiers,
     * using identical folds (so results are paired). Aggregates confusion matrices,
     * per-run accuracies, and McNemar discordant counts.
     *
     * @param clsA  classifier A (e.g. RandomForest); this method will make copies
     *              internally
     * @param dataA dataset for classifier A (must have class index set)
     * @param clsB  classifier B (e.g. J48)
     * @param dataB dataset for classifier B (must have class index set)
     * @param folds number of folds
     * @param runs  repeated runs
     * @param seed  base seed
     * @return RepeatedCVResult
     * @throws Exception
     */
    public static RepeatedCVResult repeatedPairedCrossValidation(
            Classifier clsA, Instances dataA,
            Classifier clsB, Instances dataB,
            int folds, int runs, long seed) throws Exception {

        if (dataA.numInstances() != dataB.numInstances())
            throw new IllegalArgumentException("Both datasets must have same number of instances and same ordering.");

        int n = dataA.numInstances();
        int numClasses = dataA.numClasses();
        RepeatedCVResult result = new RepeatedCVResult(numClasses);
        result.classNames = new String[numClasses];
        for (int i = 0; i < numClasses; i++)
            result.classNames[i] = dataA.classAttribute().value(i);

        result.totalInstances = 0;
        result.correctA = 0;
        result.correctB = 0;
        result.mcnemar_b = 0;
        result.mcnemar_c = 0;

        Random baseRand = new Random(seed);

        for (int run = 0; run < runs; run++) {
            // create a shuffled permutation of indices for this run (so we can ensure
            // identical splits for both datasets)
            List<Integer> idx = new ArrayList<>();
            for (int i = 0; i < n; i++)
                idx.add(i);
            long runSeed = seed + run;
            Collections.shuffle(idx, new Random(runSeed));

            // distribute indices into folds (round-robin after shuffle ensures approximate
            // stratification)
            List<List<Integer>> foldIdx = new ArrayList<>();
            for (int f = 0; f < folds; f++)
                foldIdx.add(new ArrayList<>());
            for (int i = 0; i < idx.size(); i++)
                foldIdx.get(i % folds).add(idx.get(i));

            double correctThisRunA = 0;
            double correctThisRunB = 0;

            for (int f = 0; f < folds; f++) {
                // build train/test for A and B using same indices
                Instances trainA = new Instances(dataA, 0);
                Instances testA = new Instances(dataA, 0);
                Instances trainB = new Instances(dataB, 0);
                Instances testB = new Instances(dataB, 0);

                // test indices for this fold
                Set<Integer> testSet = new HashSet<>(foldIdx.get(f));

                for (int i = 0; i < n; i++) {
                    if (testSet.contains(i)) {
                        testA.add(dataA.instance(i));
                        testB.add(dataB.instance(i));
                    } else {
                        trainA.add(dataA.instance(i));
                        trainB.add(dataB.instance(i));
                    }
                }

                // Make fresh classifier copies
                Classifier aCopy = AbstractClassifier.makeCopy(clsA);
                Classifier bCopy = AbstractClassifier.makeCopy(clsB);

                long tStartA = System.currentTimeMillis();
                aCopy.buildClassifier(trainA);
                long tEndA = System.currentTimeMillis();
                result.totalTimeMsA += (tEndA - tStartA);

                long tStartB = System.currentTimeMillis();
                bCopy.buildClassifier(trainB);
                long tEndB = System.currentTimeMillis();
                result.totalTimeMsB += (tEndB - tStartB);

                // Evaluate on this fold's test instances (manually collect predictions to
                // ensure alignment)
                for (int ti = 0; ti < testA.numInstances(); ti++) {
                    Instance instA = testA.instance(ti);
                    Instance instB = testB.instance(ti); // same row but different attributes
                    int actual = (int) instA.classValue();

                    double predA = aCopy.classifyInstance(instA);
                    double predB = bCopy.classifyInstance(instB);

                    // update confusion matrices
                    int pA = (int) predA;
                    int pB = (int) predB;
                    result.confA[actual][pA]++;
                    result.confB[actual][pB]++;

                    boolean okA = (pA == actual);
                    boolean okB = (pB == actual);
                    if (okA)
                        result.correctA++;
                    if (okB)
                        result.correctB++;
                    result.totalInstances++;

                    if (okA && !okB)
                        result.mcnemar_b++;
                    if (!okA && okB)
                        result.mcnemar_c++;
                }

                // compute run accuracies incrementally
                // We'll compute run-level accuracy after all folds for this run
            } // end folds

            // compute run accuracy for A and B
            double runAccA = (double) result.correctA / result.totalInstances;
            double runAccB = (double) result.correctB / result.totalInstances;

            // Recompute this run's counts only:
            int runCorrectA = 0, runCorrectB = 0, runTotal = 0;
            // regenerate permutation and folds for this run again
            List<Integer> idxRun = new ArrayList<>();
            for (int i = 0; i < n; i++)
                idxRun.add(i);
            Collections.shuffle(idxRun, new Random(seed + run));
            List<List<Integer>> foldIdxRun = new ArrayList<>();
            for (int f = 0; f < folds; f++)
                foldIdxRun.add(new ArrayList<>());
            for (int i = 0; i < idxRun.size(); i++)
                foldIdxRun.get(i % folds).add(idxRun.get(i));

            for (int f = 0; f < folds; f++) {
                Instances trainA = new Instances(dataA, 0);
                Instances testA = new Instances(dataA, 0);
                Instances trainB = new Instances(dataB, 0);
                Instances testB = new Instances(dataB, 0);

                Set<Integer> testSet = new HashSet<>(foldIdxRun.get(f));
                for (int i = 0; i < n; i++) {
                    if (testSet.contains(i)) {
                        testA.add(dataA.instance(i));
                        testB.add(dataB.instance(i));
                    } else {
                        trainA.add(dataA.instance(i));
                        trainB.add(dataB.instance(i));
                    }
                }
                Classifier aCopy = AbstractClassifier.makeCopy(clsA);
                Classifier bCopy = AbstractClassifier.makeCopy(clsB);
                aCopy.buildClassifier(trainA);
                bCopy.buildClassifier(trainB);
                for (int ti = 0; ti < testA.numInstances(); ti++) {
                    Instance instA = testA.instance(ti);
                    Instance instB = testB.instance(ti);
                    int actual = (int) instA.classValue();
                    int pA = (int) aCopy.classifyInstance(instA);
                    int pB = (int) bCopy.classifyInstance(instB);
                    if (pA == actual)
                        runCorrectA++;
                    if (pB == actual)
                        runCorrectB++;
                    runTotal++;
                }
            }

            double runAccA2 = (double) runCorrectA / runTotal;
            double runAccB2 = (double) runCorrectB / runTotal;
            result.runAccuraciesA.add(runAccA2);
            result.runAccuraciesB.add(runAccB2);

            // Note: we already aggregated confusion matrices and mcnemar counts above.

        } // end runs

        return result;
    }

    // Helper: compute Cohen's kappa from confusion matrix
    public static double computeKappa(int[][] conf, int total) {
        int k = conf.length;
        double po = 0;
        double[] rowSum = new double[k];
        double[] colSum = new double[k];
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                if (i == j)
                    po += conf[i][j];
                rowSum[i] += conf[i][j];
                colSum[j] += conf[i][j];
            }
        }
        po = po / total;
        double pe = 0;
        for (int i = 0; i < k; i++) {
            pe += (rowSum[i] * colSum[i]);
        }
        pe = pe / ((double) total * (double) total);
        if (1 - pe == 0)
            return 0;
        return (po - pe) / (1 - pe);
    }

    public static void printConfMatrix(int[][] conf, String[] classNames) {
        int k = conf.length;
        System.out.print("      ");
        for (int j = 0; j < k; j++)
            System.out.printf("%8s", classNames[j]);
        System.out.println();
        for (int i = 0; i < k; i++) {
            System.out.printf("%-6s", classNames[i]);
            for (int j = 0; j < k; j++)
                System.out.printf("%8d", conf[i][j]);
            System.out.println();
        }
    }

    public static void printPerClassPRF(int[][] conf, String[] classNames) {
        int k = conf.length;
        for (int c = 0; c < k; c++) {
            int tp = conf[c][c];
            int fn = 0;
            int fp = 0;
            for (int j = 0; j < k; j++) {
                if (j != c) {
                    fn += conf[c][j]; // row c not predicted as c
                    fp += conf[j][c]; // column c predicted when actual j
                }
            }
            double prec = tp + fp == 0 ? 0.0 : (double) tp / (tp + fp);
            double rec = tp + fn == 0 ? 0.0 : (double) tp / (tp + fn);
            double f1 = (prec + rec == 0) ? 0.0 : 2.0 * (prec * rec) / (prec + rec);
            System.out.printf("Class %-10s Precision=%.4f  Recall=%.4f  F1=%.4f%n", classNames[c], prec, rec, f1);
        }
    }

    // --------------------------
    // === Original helper methods (unchanged) =======
    // --------------------------

    public static Instances loadCSV(String path) {
        try {
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(path));
            Instances data = loader.getDataSet();
            return data;
        } catch (IOException ex) {
            System.err.println("Error loading CSV: " + ex.getMessage());
            return null;
        }
    }

    // Print sample predictions: actual, predicted, probability distribution
    public static void printSamplePredictions(Classifier cls, Instances data, int n) {
        try {
            System.out.println("\nSample predictions (first " + Math.min(n, data.numInstances()) + " rows):");
            System.out.printf("%4s %-12s %-12s %s%n", "idx", "actual", "predicted", "distribution");
            for (int i = 0; i < Math.min(n, data.numInstances()); i++) {
                Instance inst = data.instance(i);
                String actual = inst.classIsMissing() ? "?" : inst.classAttribute().value((int) inst.classValue());
                double predIdx = cls.classifyInstance(inst);
                String pred = inst.classAttribute().value((int) predIdx);
                double[] dist = cls.distributionForInstance(inst);
                System.out.printf("%4d %-12s %-12s %s%n", i, actual, pred, Arrays.toString(dist));
            }
        } catch (Exception e) {
            System.out.println("Sample prediction printing failed: " + e.getMessage());
        }
    }

    public static Map<String, String> buildNameMap(Instances data) {
        Map<String, String> map = new LinkedHashMap<>();
        for (int i = 0; i < data.numAttributes(); i++) {
            String orig = data.attribute(i).name();
            String key = orig.trim().toLowerCase();
            map.put(key, orig);
        }
        return map;
    }

    /**
     * Precise column finder: prefer exact 'life ladder' variants; avoid generic
     * 'life'
     */
    public static String findColumnNamePrecise(Instances data, String... candidates) {
        Map<String, String> normalizedToOrig = new HashMap<>();
        for (int i = 0; i < data.numAttributes(); i++) {
            String orig = data.attribute(i).name();
            String norm = normalizeToken(orig);
            normalizedToOrig.put(norm, orig);
        }
        for (String c : candidates) {
            String cn = normalizeToken(c);
            if (normalizedToOrig.containsKey(cn))
                return normalizedToOrig.get(cn);
        }
        for (String c : candidates) {
            String lowerc = c.trim().toLowerCase();
            for (String orig : normalizedToOrig.values()) {
                if (orig.toLowerCase().contains(lowerc) && !lowerc.trim().equals("life")) {
                    return orig;
                }
            }
        }
        return null;
    }

    public static String normalizeToken(String s) {
        return s.replaceAll("[^A-Za-z0-9]", "").toLowerCase();
    }

    // Remove attributes that are completely empty
    public static Instances removeEmptyAttributes(Instances data) throws Exception {
        ArrayList<Integer> removeIdx = new ArrayList<>();
        for (int a = 0; a < data.numAttributes(); a++) {
            Attribute attr = data.attribute(a);
            boolean allEmpty = true;
            for (int i = 0; i < data.numInstances(); i++) {
                Instance inst = data.instance(i);
                if (!inst.isMissing(attr)) {
                    if (attr.isString() || attr.isNominal()) {
                        String s = inst.toString(attr).trim();
                        if (!s.isEmpty()) {
                            allEmpty = false;
                            break;
                        }
                    } else {
                        allEmpty = false;
                        break;
                    }
                }
            }
            if (allEmpty)
                removeIdx.add(a);
        }

        if (removeIdx.isEmpty()) {
            System.out.println("No completely empty attributes found.");
            return data;
        }

        String indices = removeIdx.stream()
                .map(idx -> Integer.toString(idx + 1))
                .collect(Collectors.joining(","));
        Remove rem = new Remove();
        rem.setAttributeIndices(indices);
        rem.setInputFormat(data);
        Instances newData = Filter.useFilter(data, rem);
        System.out.println("Removed empty attributes: " + indices);
        return newData;
    }

    // Remove exact duplicate instances
    public static Instances removeDuplicateInstances(Instances data) {
        Set<String> seen = new HashSet<>();
        ArrayList<Integer> toRemove = new ArrayList<>();
        for (int i = 0; i < data.numInstances(); i++) {
            Instance inst = data.instance(i);
            StringBuilder key = new StringBuilder();
            for (int a = 0; a < data.numAttributes(); a++) {
                key.append(inst.toString(a)).append("|");
            }
            String k = key.toString();
            if (seen.contains(k))
                toRemove.add(i);
            else
                seen.add(k);
        }
        if (toRemove.isEmpty()) {
            System.out.println("No duplicate instances found.");
            return data;
        }
        Instances copy = new Instances(data);
        Collections.sort(toRemove, Collections.reverseOrder());
        for (int idx : toRemove)
            copy.delete(idx);
        System.out.println("Removed " + toRemove.size() + " duplicate instances.");
        return copy;
    }

    // Winsorize numeric attributes using IQR 1.5 rule (cap values to lower/upper)
    public static Instances winsorizeNumericIQR(Instances data) {
        System.out.println("Winsorizing numeric attributes with IQR 1.5 rule...");
        Instances copy = new Instances(data);
        for (int a = 0; a < copy.numAttributes(); a++) {
            Attribute attr = copy.attribute(a);
            if (!attr.isNumeric())
                continue;
            ArrayList<Double> vals = new ArrayList<>();
            for (int i = 0; i < copy.numInstances(); i++) {
                Instance inst = copy.instance(i);
                if (!inst.isMissing(a))
                    vals.add(inst.value(a));
            }
            if (vals.size() < 4)
                continue;
            Collections.sort(vals);
            double q1 = percentile(vals, 25);
            double q3 = percentile(vals, 75);
            double iqr = q3 - q1;
            double lower = q1 - 1.5 * iqr;
            double upper = q3 + 1.5 * iqr;
            int nClipped = 0;
            for (int i = 0; i < copy.numInstances(); i++) {
                Instance inst = copy.instance(i);
                if (inst.isMissing(a))
                    continue;
                double v = inst.value(a);
                if (v < lower) {
                    inst.setValue(a, lower);
                    nClipped++;
                } else if (v > upper) {
                    inst.setValue(a, upper);
                    nClipped++;
                }
            }
            if (nClipped > 0) {
                System.out.printf(" - Attr '%s': lower=%.4f upper=%.4f clipped=%d%n", attr.name(), lower, upper,
                        nClipped);
            }
        }
        return copy;
    }

    public static double percentile(List<Double> sorted, double p) {
        if (sorted.isEmpty())
            return Double.NaN;
        double pos = (p / 100.0) * (sorted.size() + 1);
        if (pos <= 1)
            return sorted.get(0);
        if (pos >= sorted.size())
            return sorted.get(sorted.size() - 1);
        double lower = sorted.get((int) pos - 1);
        double upper = sorted.get((int) pos);
        double frac = pos - Math.floor(pos);
        return lower + frac * (upper - lower);
    }

    // Replace missing values using Weka ReplaceMissingValues filter
    public static Instances replaceMissingValues(Instances data) throws Exception {
        ReplaceMissingValues rmv = new ReplaceMissingValues();
        rmv.setInputFormat(data);
        Instances newData = Filter.useFilter(data, rmv);
        System.out.println("Replaced missing values (numeric => mean, nominal => mode).");
        return newData;
    }

    // Add happiness_level nominal attribute (Low/Medium/High) based on life_ladder
    // quantiles; then remove original life_ladder
    public static Instances addHappinessLevelFromLifeLadder(Instances data, String lifeLadderName) throws Exception {
        Attribute lifeAttr = data.attribute(lifeLadderName);
        if (lifeAttr == null || !lifeAttr.isNumeric()) {
            throw new IllegalArgumentException("life_ladder attribute missing or not numeric: " + lifeLadderName);
        }

        // collect non-missing values
        ArrayList<Double> vals = new ArrayList<>();
        for (int i = 0; i < data.numInstances(); i++) {
            Instance inst = data.instance(i);
            if (!inst.isMissing(lifeAttr))
                vals.add(inst.value(lifeAttr));
        }
        Collections.sort(vals);
        double q33 = percentile(vals, 33);
        double q66 = percentile(vals, 66);
        System.out.printf("life_ladder quantiles: 33%%=%.4f  66%%=%.4f%n", q33, q66);

        // create nominal attribute happiness_level
        ArrayList<String> labels = new ArrayList<>(Arrays.asList("Low", "Medium", "High"));
        Attribute happiness = new Attribute("happiness_level", labels);

        // copy dataset and append new attribute at end
        Instances copy = new Instances(data);
        copy.insertAttributeAt(happiness, copy.numAttributes()); // appended at end
        int newIdx = copy.numAttributes() - 1;

        // set values
        for (int i = 0; i < copy.numInstances(); i++) {
            Instance inst = copy.instance(i);
            if (inst.isMissing(copy.attribute(lifeLadderName))) {
                inst.setValue(newIdx, "Medium");
            } else {
                double v = inst.value(copy.attribute(lifeLadderName));
                if (v <= q33)
                    inst.setValue(newIdx, "Low");
                else if (v <= q66)
                    inst.setValue(newIdx, "Medium");
                else
                    inst.setValue(newIdx, "High");
            }
        }

        // now remove the original life_ladder attribute by index (in the copy)
        int removeIndex = copy.attribute(lifeLadderName).index() + 1; // 1-based for Remove
        Remove rem = new Remove();
        rem.setAttributeIndices(Integer.toString(removeIndex));
        rem.setInputFormat(copy);
        Instances finalData = Filter.useFilter(copy, rem);

        System.out.println("Added 'happiness_level' attribute and removed '" + lifeLadderName + "'.");
        return finalData;
    }

    // Save ARFF
    public static void saveArff(Instances data, String filename) throws IOException {
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(filename));
        saver.writeBatch();
    }

    // Build latest-per-country dataset
    public static Instances buildLatestPerCountry(Instances data) {
        int countryIdx = -1;
        int yearIdx = -1;
        for (int a = 0; a < data.numAttributes(); a++) {
            String name = data.attribute(a).name().toLowerCase();
            if (countryIdx == -1 && name.contains("country"))
                countryIdx = a;
            if (yearIdx == -1 && name.contains("year"))
                yearIdx = a;
        }
        if (countryIdx == -1 || yearIdx == -1) {
            System.out.println(
                    "Cannot build latest-per-country: country/year attribute not found. Returning full dataset.");
            return new Instances(data);
        }

        Map<String, Integer> bestIdx = new HashMap<>();
        Map<String, Double> bestYear = new HashMap<>();
        for (int i = 0; i < data.numInstances(); i++) {
            Instance inst = data.instance(i);
            String country = inst.isMissing(countryIdx) ? ("__MISSING__" + i) : inst.stringValue(countryIdx);
            double year = inst.isMissing(yearIdx) ? Double.NEGATIVE_INFINITY : inst.value(yearIdx);
            if (!bestIdx.containsKey(country) || year > bestYear.get(country)) {
                bestIdx.put(country, i);
                bestYear.put(country, year);
            }
        }

        Instances out = new Instances(data, bestIdx.size());
        for (Integer idx : bestIdx.values())
            out.add((Instance) data.instance(idx).copy());
        System.out.println("Built latest-per-country dataset with " + out.numInstances() + " rows.");
        return out;
    }

    // Standardize numeric attributes
    public static Instances standardizeIfNeeded(Instances data) {
        try {
            Standardize std = new Standardize();
            std.setInputFormat(data);
            Instances out = Filter.useFilter(data, std);
            System.out.println("Standardized copy created for algorithms that require scaling.");
            return out;
        } catch (Exception e) {
            System.out.println("Standardize failed: " + e.getMessage());
            return data;
        }
    }

    // Add cluster attribute using AddCluster
    public static Instances addClusterAttribute(Instances data, int k) {
        try {
            SimpleKMeans km = new SimpleKMeans();
            km.setNumClusters(k);
            km.setSeed((int) RANDOM_SEED);
            km.setPreserveInstancesOrder(true);

            AddCluster addCluster = new AddCluster();
            addCluster.setClusterer(km);
            addCluster.setInputFormat(data);
            Instances out = Filter.useFilter(data, addCluster);
            System.out.println("Added cluster id attribute (k=" + k + ").");
            return out;
        } catch (Exception e) {
            System.out.println("AddCluster failed: " + e.getMessage());
            return data;
        }
    }

    public static void printComparisonSummary(Evaluation evalA, double timeAms, Evaluation evalB, double timeBms) {
        System.out.println("\n=== Comparison Summary ===");
        System.out.printf("Method A (RandomForest): Accuracy=%.4f, Kappa=%.4f, Time(ms)=%.0f%n",
                (1 - evalA.errorRate()), evalA.kappa(), timeAms);
        System.out.printf("Method B (J48 + AddCluster as predictor): Accuracy=%.4f, Kappa=%.4f, Time(ms)=%.0f%n",
                (1 - evalB.errorRate()), evalB.kappa(), timeBms);
        System.out.println("Refer to per-class details above for Precision/Recall/F1 and confusion matrices.");
    }

    // Visualize clusters on numeric features with KMeans (Swing window)
    // NOTE: this re-runs KMeans on numeric-only copy; kept for backward
    // compatibility.
    public static void visualizeClustersOnNumericFeatures(Instances data, int k) {
        try {
            ArrayList<Attribute> attrs = new ArrayList<>();
            ArrayList<Integer> numericIdx = new ArrayList<>();
            for (int a = 0; a < data.numAttributes(); a++) {
                if (data.attribute(a).isNumeric()) {
                    attrs.add((Attribute) data.attribute(a).copy());
                    numericIdx.add(a);
                }
            }

            if (attrs.isEmpty()) {
                System.out.println("No numeric attributes for visualization. Skipping.");
                return;
            }

            Instances numeric = new Instances("numeric_copy", attrs, data.numInstances());
            for (int i = 0; i < data.numInstances(); i++) {
                Instance orig = data.instance(i);
                double[] vals = new double[attrs.size()];
                for (int j = 0; j < numericIdx.size(); j++) {
                    int origIdx = numericIdx.get(j);
                    if (orig.isMissing(origIdx))
                        vals[j] = Double.NaN;
                    else
                        vals[j] = orig.value(origIdx);
                }
                numeric.add(new DenseInstance(1.0, vals));
            }

            SimpleKMeans kmeans = new SimpleKMeans();
            kmeans.setNumClusters(k);
            kmeans.setSeed((int) RANDOM_SEED);
            kmeans.setPreserveInstancesOrder(true);
            kmeans.buildClusterer(numeric);
            int[] assigns = kmeans.getAssignments();

            ArrayList<String> clusterVals = new ArrayList<>();
            for (int i = 0; i < k; i++)
                clusterVals.add(Integer.toString(i));
            Attribute clusterAttr = new Attribute("cluster", clusterVals);
            numeric.insertAttributeAt(clusterAttr, numeric.numAttributes());

            for (int i = 0; i < numeric.numInstances(); i++) {
                if (i < assigns.length) {
                    numeric.instance(i).setValue(numeric.numAttributes() - 1, Integer.toString(assigns[i]));
                } else {
                    numeric.instance(i).setValue(numeric.numAttributes() - 1, "0");
                }
            }

            PlotData2D plotData = new PlotData2D(numeric);
            plotData.setPlotName("KMeans clusters (numeric features)");
            plotData.addInstanceNumberAttribute();

            EventQueue.invokeLater(() -> {
                try {
                    VisualizePanel vp = new VisualizePanel();
                    vp.setName("KMeans cluster visualization");
                    vp.addPlot(plotData);

                    JFrame jf = new JFrame("KMeans clusters (numeric features)");
                    jf.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
                    jf.setLayout(new BorderLayout());
                    jf.add(vp, BorderLayout.CENTER);
                    jf.setSize(900, 600);
                    jf.setLocationRelativeTo(null);
                    jf.setVisible(true);
                } catch (Exception ex) {
                    ex.printStackTrace();
                }
            });

            System.out.println("Cluster visualization launched (window should appear).");

        } catch (Exception e) {
            System.out.println("Cluster visualization failed: " + e.getMessage());
        }
    }

    // Visualize numeric attributes colored by the cluster attribute that AddCluster
    // appended.
    // This uses the actual nominal cluster attribute in 'dataWithCluster' (no
    // re-running KMeans).
    public static void visualizeClustersUsingClusterAttr(Instances dataWithCluster, int k) {
        try {
            // find index of cluster attribute (it is nominal, name likely "cluster" or
            // similar)
            int clusterIdx = -1;
            for (int a = 0; a < dataWithCluster.numAttributes(); a++) {
                Attribute att = dataWithCluster.attribute(a);
                if (att.isNominal() && att.name().toLowerCase().contains("cluster")) {
                    clusterIdx = a;
                    break;
                }
            }
            if (clusterIdx == -1) {
                System.out.println(
                        "No 'cluster' nominal attribute found — falling back to numeric KMeans visualization.");
                visualizeClustersOnNumericFeatures(dataWithCluster, k);
                return;
            }

            // Build an Instances object with numeric attributes only (keep cluster attr
            // last)
            ArrayList<Attribute> attrs = new ArrayList<>();
            ArrayList<Integer> numericIdx = new ArrayList<>();
            for (int a = 0; a < dataWithCluster.numAttributes(); a++) {
                if (a == clusterIdx)
                    continue;
                if (dataWithCluster.attribute(a).isNumeric()) {
                    attrs.add((Attribute) dataWithCluster.attribute(a).copy());
                    numericIdx.add(a);
                }
            }
            // copy cluster nominal attribute (so we can color by it)
            attrs.add((Attribute) dataWithCluster.attribute(clusterIdx).copy());

            Instances numericWithCluster = new Instances("numeric_with_cluster", attrs, dataWithCluster.numInstances());

            for (int i = 0; i < dataWithCluster.numInstances(); i++) {
                Instance orig = dataWithCluster.instance(i);
                double[] vals = new double[attrs.size()];
                for (int j = 0; j < numericIdx.size(); j++) {
                    int origIdx = numericIdx.get(j);
                    if (orig.isMissing(origIdx))
                        vals[j] = Double.NaN;
                    else
                        vals[j] = orig.value(origIdx);
                }
                // set cluster nominal value
                String clv = orig.stringValue(clusterIdx);
                Attribute newClusterAttr = numericWithCluster.attribute(numericWithCluster.numAttributes() - 1);
                int nomIndex = newClusterAttr.indexOfValue(clv);
                if (nomIndex == -1)
                    nomIndex = 0;
                vals[vals.length - 1] = nomIndex;
                numericWithCluster.add(new DenseInstance(1.0, vals));
            }

            PlotData2D plotData = new PlotData2D(numericWithCluster);
            plotData.setPlotName("Clusters (cluster attribute present)");
            plotData.addInstanceNumberAttribute();

            EventQueue.invokeLater(() -> {
                try {
                    VisualizePanel vp = new VisualizePanel();
                    vp.setName("Clusters (from AddCluster attribute)");
                    vp.addPlot(plotData);

                    JFrame jf = new JFrame("Clusters (from AddCluster attribute)");
                    jf.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
                    jf.setLayout(new BorderLayout());
                    jf.add(vp, BorderLayout.CENTER);
                    jf.setSize(900, 600);
                    jf.setLocationRelativeTo(null);
                    jf.setVisible(true);
                } catch (Exception ex) {
                    ex.printStackTrace();
                }
            });
            System.out.println("Cluster visualization (from cluster attribute) launched.");
        } catch (Exception e) {
            System.out.println("Cluster visualization (from cluster attr) failed: " + e.getMessage());
        }
    }

    // -------------------------
    // === Data analysis helpers
    // -------------------------

    // Print missing count and percent per column
    public static void printMissingSummary(Instances data) {
        System.out.println("\nMissing values per column:");
        System.out.printf("%-40s %10s %10s%n", "Attribute", "Missing", "Percent");
        for (int a = 0; a < data.numAttributes(); a++) {
            int miss = 0;
            for (int i = 0; i < data.numInstances(); i++)
                if (data.instance(i).isMissing(a))
                    miss++;
            double pct = 100.0 * miss / data.numInstances();
            System.out.printf("%-40s %10d %9.3f%%%n", data.attribute(a).name(), miss, pct);
        }
    }

    // Print numeric summary (count, mean, std, min, 25,50,75,max)
    public static void printNumericSummary(Instances data) {
        System.out.println("\nNumeric summary (count, mean, std, min, 25%, 50%, 75%, max):");
        for (int a = 0; a < data.numAttributes(); a++) {
            Attribute attr = data.attribute(a);
            if (!attr.isNumeric())
                continue;
            ArrayList<Double> vals = new ArrayList<>();
            for (int i = 0; i < data.numInstances(); i++) {
                Instance inst = data.instance(i);
                if (!inst.isMissing(a))
                    vals.add(inst.value(a));
            }
            if (vals.isEmpty())
                continue;
            Collections.sort(vals);
            double count = vals.size();
            double sum = 0;
            double min = vals.get(0), max = vals.get(vals.size() - 1);
            for (double v : vals)
                sum += v;
            double mean = sum / count;
            double sd = 0;
            for (double v : vals)
                sd += (v - mean) * (v - mean);
            sd = Math.sqrt(sd / count);
            System.out.printf("%-40s %7.0f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f%n",
                    attr.name(), count, mean, sd, min, percentile(vals, 25), percentile(vals, 50), percentile(vals, 75),
                    max);
        }
    }

    // Print Pearson correlation matrix for numeric attributes (simple)
    public static void printCorrelationMatrix(Instances data) {
        System.out.println("\nCorrelation matrix (numeric attributes):");
        List<Integer> numIdx = new ArrayList<>();
        List<String> names = new ArrayList<>();
        for (int a = 0; a < data.numAttributes(); a++) {
            if (data.attribute(a).isNumeric()) {
                numIdx.add(a);
                names.add(data.attribute(a).name());
            }
        }
        int n = numIdx.size();
        if (n == 0) {
            System.out.println("No numeric attributes.");
            return;
        }

        // compute means and stds
        double[] mean = new double[n];
        double[] sd = new double[n];
        double[][] vals = new double[n][data.numInstances()];
        for (int j = 0; j < n; j++) {
            int a = numIdx.get(j);
            int idx = 0;
            double sum = 0;
            for (int i = 0; i < data.numInstances(); i++) {
                double v = Double.NaN;
                if (!data.instance(i).isMissing(a))
                    v = data.instance(i).value(a);
                if (Double.isNaN(v))
                    v = 0; // treat missing as 0 for correlation calc (simple)
                vals[j][idx++] = v;
                sum += v;
            }
            mean[j] = sum / data.numInstances();
            double ss = 0;
            for (int i = 0; i < data.numInstances(); i++)
                ss += (vals[j][i] - mean[j]) * (vals[j][i] - mean[j]);
            sd[j] = Math.sqrt(ss / data.numInstances());
        }

        // compute corr matrix
        System.out.print(String.format("%-25s", ""));
        for (String nm : names)
            System.out.print(String.format("%12s", nm.length() > 10 ? nm.substring(0, 10) : nm));
        System.out.println();
        for (int i = 0; i < n; i++) {
            System.out.print(
                    String.format("%-25s", names.get(i).length() > 22 ? names.get(i).substring(0, 22) : names.get(i)));
            for (int j = 0; j < n; j++) {
                double cov = 0;
                for (int t = 0; t < data.numInstances(); t++)
                    cov += (vals[i][t] - mean[i]) * (vals[j][t] - mean[j]);
                double corr = (sd[i] == 0 || sd[j] == 0) ? 0 : cov / (data.numInstances() * sd[i] * sd[j]);
                System.out.print(String.format("%12.3f", corr));
            }
            System.out.println();
        }
    }

    // Print top countries and counts by year
    public static void printCountsByYearAndTopCountries(Instances data, int topN) {
        int countryIdx = -1, yearIdx = -1;
        for (int a = 0; a < data.numAttributes(); a++) {
            String name = data.attribute(a).name().toLowerCase();
            if (countryIdx == -1 && name.contains("country"))
                countryIdx = a;
            if (yearIdx == -1 && name.contains("year"))
                yearIdx = a;
        }
        if (yearIdx != -1) {
            Map<Double, Integer> byYear = new TreeMap<>();
            for (int i = 0; i < data.numInstances(); i++) {
                Instance inst = data.instance(i);
                if (!inst.isMissing(yearIdx)) {
                    double y = inst.value(yearIdx);
                    byYear.put(y, byYear.getOrDefault(y, 0) + 1);
                }
            }
            System.out.println("\nCounts by Year:");
            for (Map.Entry<Double, Integer> e : byYear.entrySet())
                System.out.printf("  %d: %d%n", e.getKey().intValue(), e.getValue());
        }
        if (countryIdx != -1) {
            Map<String, Integer> byCountry = new HashMap<>();
            for (int i = 0; i < data.numInstances(); i++) {
                Instance inst = data.instance(i);
                String c = inst.isMissing(countryIdx) ? "__MISSING__" : inst.stringValue(countryIdx);
                byCountry.put(c, byCountry.getOrDefault(c, 0) + 1);
            }
            System.out.println("\nTop countries by row count:");
            byCountry.entrySet().stream()
                    .sorted((a, b) -> Integer.compare(b.getValue(), a.getValue()))
                    .limit(topN)
                    .forEach(e -> System.out.printf("  %-25s %4d%n", e.getKey(), e.getValue()));
        }
    }

    // Print sample table rows (first n), truncated columns for readability
    public static void printSampleTable(Instances data, int n) {
        int display = Math.min(n, data.numInstances());
        List<String> colNames = new ArrayList<>();
        for (int a = 0; a < data.numAttributes(); a++)
            colNames.add(data.attribute(a).name());
        // header
        String header = "";
        for (String cn : colNames)
            header += String.format("%-20s", cn.length() > 18 ? cn.substring(0, 18) : cn);
        System.out.println(header);
        for (int i = 0; i < display; i++) {
            StringBuilder sb = new StringBuilder();
            Instance inst = data.instance(i);
            for (int a = 0; a < data.numAttributes(); a++) {
                String s;
                if (inst.isMissing(a))
                    s = "NA";
                else
                    s = inst.attribute(a).isNumeric() ? String.format("%.4f", inst.value(a)) : inst.stringValue(a);
                if (s.length() > 18)
                    s = s.substring(0, 18);
                sb.append(String.format("%-20s", s));
            }
            System.out.println(sb.toString());
        }
        System.out.println("... (printed " + display + " rows)");
    }
}
