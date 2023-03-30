<?php

namespace Darvin\LSTM;

use Darvin\LSTM\Helpers\{MatrixHelper, Random};
use Rubix\ML\Helpers\Params;
use JetBrains\PhpStorm\ArrayShape;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\{Verbose, DataType, Estimator, EstimatorType,Learner};
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\NeuralNet\ActivationFunctions\{Sigmoid, Softmax};
use Rubix\ML\{Online, Persistable, Probabilistic};
use Rubix\ML\Specifications\{DatasetHasDimensionality, DatasetIsNotEmpty, SamplesAreCompatibleWithEstimator, SpecificationChain};
use Rubix\ML\Traits\{AutotrackRevisions, LoggerAware};
use Tensor\{Matrix, Vector};


class LSTM implements Estimator, Learner, Online, Probabilistic, Verbose, Persistable
{
    use AutotrackRevisions, LoggerAware;

    /**
     * Number of training iterations
     *
     * @var int
     */
    protected int $epochs;

    /**
     * Learning rate
     *
     * @var float
     */
    protected float $learningRate;

    /**
     * Number of units in the hidden layer
     *
     * @var int
     */
    protected int $nH;

    /**
     * Size of batch
     *
     * @var int
     */
    protected int $seqLen;

    /**
     * 1st momentum parameter
     *
     * @var float
     */
    protected float $beta1;

    /**
     * 2nd momentum parameter
     *
     * @var float
     */
    protected float $beta2;


    /**
     * @var int
     */
    protected int $vocabSize = 0;

    /**
     * LSTM Cell Parameters
     *
     * @var ?array
     */
    protected ?array $params = null;


    /**
     * Grads and adam parameters
     *
     * @var ?array
     */
    protected ?array $grads = null;
    protected array $adamParams;
    protected float $smoothLoss;

    protected array $indexToToken = [];
    protected array $tokenToIndex = [];

    protected ?Matrix $hPrev = null;
    protected ?Matrix $cPrev = null;

    public function __construct(
        int   $epochs = 200,
        float $learningRate = 0.001,
        int   $unitsInHidden = 100,
        int   $seqLen = 5,
        float $beta1 = 0.9,
        float $beta2 = 0.999
    )
    {
        if ($epochs <= 0) {
            throw new InvalidArgumentException('Number of epochs'
                . " must be greater than 0, $epochs given.");
        }

        if ($learningRate <= 0.0) {
            throw new InvalidArgumentException('Learning rate'
                . " must be greater than 0, $learningRate given.");
        }

        if ($unitsInHidden <= 0) {
            throw new InvalidArgumentException('Units in hidden layer'
                . " must be greater than 0, $unitsInHidden given.");
        }

        if ($seqLen <= 0) {
            throw new InvalidArgumentException('Sequence length'
                . " must be greater than 0, $seqLen given.");
        }

        if ($beta1 <= 0.0 || $beta2 <= 0.0) {
            throw new InvalidArgumentException('Beta 1 and Beta 2'
                . " must be greater than 0, beta1 is $beta1, beta2 is $beta2 given.");
        }

        $this->epochs = $epochs;
        $this->learningRate = $learningRate;
        $this->nH = $unitsInHidden;
        $this->seqLen = $seqLen;
        $this->beta1 = $beta1;
        $this->beta2 = $beta2;
        $this->smoothLoss = 0.0;
        $this->adamParams = [];

    }

    /**
     * @return string
     */
    public function __toString()
    {
        return 'Long short-term memory (' . Params::stringify($this->params()) . ')';
    }

    /**
     * @return array
     */
    public function params(): array
    {
        return [
            'epochs' => $this->epochs,
            'learning rate' => $this->learningRate,
            'units in hidden' => $this->nH,
            'sequence length' => $this->seqLen,
            'beta1' => $this->beta1,
            'beta2' => $this->beta2,
            'vocab size' => $this->vocabSize
        ];
    }

    /**
     * @return EstimatorType
     */
    public function type(): EstimatorType
    {
        return EstimatorType::regressor();
    }

    /**
     * @return array
     */
    public function compatibility(): array
    {
        return [
            DataType::categorical(),
        ];
    }

    /**
     * @param Dataset $dataset
     * @return array
     */
    public function predict(Dataset $dataset): array
    {
        $prediction = $this->sample($this->hPrev, $this->cPrev, 20, $dataset);
        return [$prediction];
    }

    /**
     * @param Matrix $hPrev
     * @param Matrix $cPrev
     * @param $sampleSize
     * @param Dataset|null $dataset
     * @return string
     */
    public function sample(Matrix $hPrev, Matrix $cPrev, $sampleSize, Dataset $dataset = null)
    {
        $x = Matrix::zeros($this->vocabSize, 1);

        $h = $hPrev;
        $c = $cPrev;
        $samples = $dataset?->feature(0);
        $sampleString = [];

        for ($t = 0; $t < $sampleSize; $t++) {
            $fs = $this->forwardStep($x, $h, $c);
            /** @var Matrix $yHat */
            $yHat = $fs[0];
            /** @var Matrix $h */
            $h = $fs[2];
            /** @var Matrix $c */
            $c = $fs[4];

            if ($samples && isset($samples[$t])) {
                $idx = $this->tokenToIndex[$samples[$t]];
            } else {
                $idx = Random::choice(range(0, $this->vocabSize - 1), $yHat->flatten());
            }

            $x = MatrixHelper::zeros($this->vocabSize, 1);
            // TODO: переделать так чтобы передавать векторы в forwardStep (x) а не матрицы
            MatrixHelper::setVectorAt($idx, $x, Vector::build([1]));
            $sampleString [] = $this->indexToToken[$idx];
        }
        return implode("", $sampleString);
    }

    /**
     * @param Matrix $x
     * @param Matrix $hPrev
     * @param Matrix $cPrev
     * @return array
     */
    public function forwardStep(Matrix $x, Matrix $hPrev, Matrix $cPrev): array
    {
        $sigmoid = new Sigmoid();
        $softmax = new Softmax();
        $z = MatrixHelper::vstack($hPrev, $x);


        $f = $sigmoid->activate($this->params["Wf"]
            ->matmul($z)
            ->add($this->params["bf"]));

        $i = $sigmoid->activate($this->params["Wi"]
            ->matmul($z)
            ->add($this->params["bi"]));

        $cBar = $this->params["Wc"]
            ->matmul($z)
            ->add($this->params["bc"])->map('tanh');


        $fMcPrev = $f->multiply($cPrev);
        $iMcBar = $i->multiply($cBar);

        $c = $fMcPrev->add($iMcBar);

        $o = $sigmoid->activate($this->params["Wo"]
            ->matmul($z)
            ->add($this->params["bo"]));

        $h = $o->multiply($c->map('tanh'));

        $v = $this->params["Wv"]
            ->matmul($h)
            ->add($this->params["bv"]);

        $yHat = $softmax->activate($v);

        return [$yHat, $v, $h, $o, $c, $cBar, $i, $f, $z];
    }

    /**
     * @param Dataset $dataset
     */
    public function partial(Dataset $dataset): void
    {
        if (!$this->params && !$this->grads) {
            $this->train($dataset);
            return;
        }

        SpecificationChain::with([
            new DatasetIsNotEmpty($dataset),
            new SamplesAreCompatibleWithEstimator($dataset, $this),
            new DatasetHasDimensionality($dataset, 1),
        ])->check();

        $this->logger?->info("Training $this");

        $J = [];

        $numBatches = floor($dataset->count() / $this->seqLen);
        $dataTrimmed = $dataset->slice(0, $numBatches * $this->seqLen)->feature(0);

        for ($epoch = 0; $epoch < $this->epochs; $epoch++) {
            $this->hPrev = Matrix::zeros($this->nH, 1);
            $this->cPrev = Matrix::zeros($this->nH, 1);


            for ($j = 0; $j < count($dataTrimmed) - $this->seqLen; $j += $this->seqLen) {
                $xBatch = $this->makeBatch(array_slice($dataTrimmed, $j, $j + $this->seqLen));
                $yBatch = $this->makeBatch(array_slice($dataTrimmed, $j + 1, $j + $this->seqLen + 1));

                list($loss, $this->hPrev, $this->cPrev) = $this->forwardBackward($xBatch, $yBatch, $this->hPrev, $this->cPrev);
                $this->smoothLoss = $this->smoothLoss * 0.999 + $loss * 0.001;
                $J[] = $this->smoothLoss;

                $this->clipGrads();
                $bachNum = $epoch * $this->epochs + $j / $this->seqLen + 1;
                $this->updateParams($bachNum);

                if ($this->logger) {
                    if ($j % 400000 === 0) {
                        $sample = $this->sample($this->hPrev, $this->cPrev, 12);
                        $this->logger->info(sprintf("Epoch: %s\tBatch: %s - %s\tLoss:%s\n%s\n",
                            $epoch, $j, $j + $this->seqLen, round($this->smoothLoss, 2), $sample));
                    }
                }
            }
        }
    }

    /**
     * @param Dataset $dataset
     */
    public function train(Dataset $dataset): void
    {
        SpecificationChain::with([
            new DatasetIsNotEmpty($dataset),
        ])->check();

        $this->vocabSize = $dataset->deduplicate()->count();

        $this->params = $this->initializeWeightsAndBiases($this->vocabSize, $this->nH);
        $this->initializeAdamParamsAndGrads();

        $this->tokenizeSamples($dataset);

        if (floor(($dataset->count() - $this->seqLen) / $this->seqLen) <= 0) {
            throw new InvalidArgumentException('Dataset has not enough samples for training'
                . " $this->seqLen sequence length given.");
        }

        $this->partial($dataset);
    }

    /**
     * @param int $vocabSize
     * @param int $unitsInHiddenLayer
     * @return array
     */
    #[ArrayShape([
        "Wf" => Matrix::class,
        "bf" => Matrix::class,
        "Wi" => Matrix::class,
        "bi" => Matrix::class,
        "Wc" => Matrix::class,
        "bc" => Matrix::class,
        "Wo" => Matrix::class,
        "bo" => Matrix::class,
        "Wv" => Matrix::class,
        "bv" => Matrix::class
    ])]
    public function initializeWeightsAndBiases(int $vocabSize, int $unitsInHiddenLayer): array
    {
        /**
         * initialise weights and biases
         */
        $params = [];
        $nH = $unitsInHiddenLayer;
        $std = (1.0 / sqrt($vocabSize + $nH));

        /**
         * Forget gate
         */
        $params["Wf"] = MatrixHelper::gaussian($nH, $nH + $vocabSize)->multiplyScalar($std);
        $params["bf"] = MatrixHelper::ones($nH, 1);

        /**
         * Input Gate
         */
        $params["Wi"] = MatrixHelper::gaussian($nH, $nH + $vocabSize)->multiplyScalar($std);
        $params["bi"] = MatrixHelper::zeros($nH, 1);

        /**
         * Cell Gate
         */
        $params["Wc"] = MatrixHelper::gaussian($nH, $nH + $vocabSize)->multiplyScalar($std);
        $params["bc"] = MatrixHelper::zeros($nH, 1);

        /**
         * Output Gate
         */
        $params["Wo"] = MatrixHelper::gaussian($nH, $nH + $vocabSize)->multiplyScalar($std);
        $params["bo"] = MatrixHelper::zeros($nH, 1);

        /**
         * Output
         */
        $params["Wv"] = MatrixHelper::gaussian($vocabSize, $nH)->multiplyScalar(1.0 / sqrt($vocabSize));
        $params["bv"] = MatrixHelper::zeros($vocabSize, 1);


        return $params;
    }

    /**
     * Initialise gradients and Adam parameters
     */
    public function initializeAdamParamsAndGrads()
    {
        $this->grads = [];
        $this->adamParams = [];

        foreach ($this->params as $key => $param) {
            $this->grads["d" . $key] = MatrixHelper::zerosLike($this->params[$key]);
            $this->adamParams["m" . $key] = MatrixHelper::zerosLike($this->params[$key]);
            $this->adamParams["v" . $key] = MatrixHelper::zerosLike($this->params[$key]);
        }
        $this->smoothLoss = -log(1.0 / $this->vocabSize) * $this->seqLen;
    }

    /**
     * Simply index-based encoding
     *
     * @param Dataset $dataset
     */
    public function tokenizeSamples(Dataset $dataset)
    {
        $tokens = $dataset->deduplicate()->randomize()->feature(0);
        $this->indexToToken = $tokens;
        $this->tokenToIndex = array_flip($tokens);
    }

    /**
     * @param array $data
     * @return array
     */
    public function makeBatch(array $data): array
    {
        $result = [];
        foreach ($data as $item) {
            $result[] = $this->tokenToIndex[$item];
        }
        return $result;
    }

    /**
     * @param $xBath
     * @param $yBatch
     * @param $hPrev
     * @param $cPrev
     * @return array
     */
    public function forwardBackward($xBath, $yBatch, $hPrev, $cPrev): array
    {
        $x = $z = [];
        $f = $i = $cBar = $c = $o = [];
        $yHat = $v = $h = [];

        $h[-1] = $hPrev;
        $c[-1] = $cPrev;

        $loss = 0;
        for ($t = 0; $t < $this->seqLen; $t++) {
            $m = MatrixHelper::zeros($this->vocabSize, 1);

            MatrixHelper::setVectorAt($xBath[$t], $m, Vector::build([1]));
            $x[$t] = $m;

            list($yHat[$t], $v[$t], $h[$t], $o[$t], $c[$t], $cBar[$t], $i[$t], $f[$t], $z[$t]) =
                $this->forwardStep($x[$t], $h[$t - 1], $c[$t - 1]);
            $loss += -log($yHat[$t][$yBatch[$t]][0]);
        }

        $this->resetGradients();

        $dhNext = MatrixHelper::zerosLike($h[0]);
        $dcNext = MatrixHelper::zerosLike($c[0]);

        for ($t = $this->seqLen - 1; $t >= 0; $t--) {
            list($dhNext, $dcNext) = $this->backwardStep($yBatch[$t], $yHat[$t], $dhNext,
                $dcNext, $c[$t - 1], $z[$t], $f[$t], $i[$t],
                $cBar[$t], $c[$t], $o[$t], $h[$t]);

        }
        return [$loss, $h[$this->seqLen - 1], $c[$this->seqLen - 1]];

    }

    /**
     * Resets gradients to zero before each backpropagation
     */
    public function resetGradients()
    {
        foreach ($this->grads as $key => $grad) {
            $this->grads[$key] = MatrixHelper::zerosLike($grad);
        }
    }

    /**
     * @param int $y
     * @param Matrix $yHat
     * @param Matrix $dhNext
     * @param Matrix $dcNext
     * @param Matrix $cPrev
     * @param Matrix $z
     * @param Matrix $f
     * @param Matrix $i
     * @param Matrix $cBar
     * @param Matrix $c
     * @param Matrix $o
     * @param Matrix $h
     * @return array
     */
    public function backwardStep(
        int    $y,
        Matrix $yHat,
        Matrix $dhNext,
        Matrix $dcNext,
        Matrix $cPrev,
        Matrix $z,
        Matrix $f,
        Matrix $i,
        Matrix $cBar,
        Matrix $c,
        Matrix $o,
        Matrix $h
    ): array
    {
        $dv = $yHat;
        MatrixHelper::setVectorAt($y, $dv, $dv->offsetGet($y)->subtractScalar(1));

        $this->grads["dWv"] = $this->grads["dWv"]->add($dv->matmul($h->transpose()));
        $this->grads["dbv"] = $this->grads["dbv"]->add($dv);


        $dh = $this->params["Wv"]->transpose()->matmul($dv);
        $dh = $dh->add($dhNext);

        $do = $dh->multiply($c->map("tanh"));

        $daO = MatrixHelper::onesLike($o)->subtract($o)->multiply($o)->multiply($do);

        $this->grads["dWo"] = $this->grads["dWo"]->add($daO->matmul($z->transpose()));
        $this->grads["dbo"] = $this->grads["dbo"]->add($daO);
        $dc = MatrixHelper::onesLike($c)->subtract($c->map("tanh")->pow(2))->multiply($o)->multiply($dh);

        $dc = $dc->add($dcNext);

        $dcBar = $dc->multiply($i);

        $daC = $dcBar->multiply(MatrixHelper::onesLike($cBar)->subtract($cBar->pow(2)));

        $this->grads["dWc"] = $this->grads["dWc"]->add($daC->matmul($z->transpose()));
        $this->grads["dbc"] = $this->grads["dbc"]->add($daC);

        $di = $dc->multiply($cBar);
        $daI = MatrixHelper::onesLike($i)->subtract($i)->multiply($i)->multiply($di);

        $this->grads["dWi"] = $this->grads["dWi"]->add($daI->matmul($z->transpose()));
        $this->grads["dbi"] = $this->grads["dbi"]->add($daI);

        $df = $dc->multiply($cPrev);
        $daF = MatrixHelper::onesLike($f)->subtract($f)->multiply($f)->multiply($df);
        $this->grads["dWf"] = $this->grads["dWf"]->add($daF->matmul($z->transpose()));
        $this->grads["dbf"] = $this->grads["dbf"]->add($daF);


        $dz = $this->params["Wf"]->transpose()->matmul($daF)
            ->add($this->params["Wi"]->transpose()->matmul($daI))
            ->add($this->params["Wc"]->transpose()->matmul($daC))
            ->add($this->params["Wo"]->transpose()->matmul($daO));

        $dhPrev = MatrixHelper::slice($dz, 0, $this->nH);
        $dcPrev = $f->multiply($dc);
        return [$dhPrev, $dcPrev];
    }

    /**
     *
     */
    public function clipGrads()
    {
        foreach ($this->grads as $key => $grad) {
            $this->grads[$key] = $this->grads[$key]->clip(-5, 5);
        }
    }

    /**
     * @param $batchNum
     */
    public function updateParams($batchNum)
    {

        foreach ($this->params as $key => $param) {

            $this->adamParams["m" . $key] = $this->adamParams["m" . $key]
                ->multiply($this->beta1)
                ->add(
                    $this->grads["d" . $key]->multiply(1 - $this->beta1)
                );

            $this->adamParams["v" . $key] = $this->adamParams["v" . $key]
                ->multiply($this->beta2)
                ->add(
                    $this->grads["d" . $key]->pow(2)->multiply(1 - $this->beta2)
                );

            /** @var Matrix $mCorrelated */
            $mCorrelated = $this->adamParams["m" . $key]->divide(1 - $this->beta1 ** $batchNum);
            /** @var Matrix $vCorrelated */
            $vCorrelated = $this->adamParams["v" . $key]->divide(1 - $this->beta2 ** $batchNum);

            $re = $mCorrelated->multiply($this->learningRate)->divide($vCorrelated->sqrt()->add(1e-8));
            $this->params[$key] = $this->params[$key]->subtract($re);
        }
    }

    public function proba(Dataset $dataset): array
    {
        // TODO: Implement proba() method.
    }

    public function trained(): bool
    {
        //TODO:
        return true;
    }
}