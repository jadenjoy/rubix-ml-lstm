<?php

use Darvin\LSTM\LSTM;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Loggers\Screen;
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Pipeline;
use Rubix\ML\Transformers\TextNormalizer;

require_once 'vendor/autoload.php';

ini_set('memory_limit', '-1');

/**
 * Dataset loading
 */
$extractor = new CSV('dataset/NationalNames.csv', true, ',', '"');
$dataset = Labeled::fromIterator($extractor)
    ->head(n:1000)
    ->feature(offset: 1);

$dataset = Unlabeled::build(str_split(implode(" ",$dataset)));

$logger = new Screen();
$logger->info("Welcome to LSTM");

$estimator = new PersistentModel(
    new Pipeline([
        new TextNormalizer(),
    ], new LSTM(300, 0.001, 30)),
    new Filesystem('names.rbx', true)
);

$estimator->setLogger($logger);
$estimator->train($dataset);

if (strtolower(trim(readline('Save this model? (y|[n]): '))) === 'y') {
    $estimator->save();
}

