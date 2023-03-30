<?php
include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Datasets\Unlabeled;

ini_set('memory_limit', '-1');

$estimator = PersistentModel::load(new Filesystem('names.rbx'));

while (empty($text)) $text = readline("Enter start of the name:\n");

$dataset = new Unlabeled(
    str_split(strtolower($text))
);

$prediction = current($estimator->predict($dataset));

echo "The name is: $prediction" . PHP_EOL;