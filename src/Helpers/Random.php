<?php

namespace Darvin\LSTM\Helpers;

use Tensor\Vector;

class Random
{
    /**
     * @param array|Vector $stack
     * @param array|Vector $prob
     * @return mixed|null
     */
    public static function choice(array|Vector $stack, array|Vector $prob): mixed
    {
        $stack = is_array($stack) ? VectorHelper::build($stack) : $stack;
        $prob = is_array($prob) ? VectorHelper::build($prob) : $prob;

        $totalProbability = $prob->sum();
        $stopAt = Distribution::uniform(0, $totalProbability);
        $currentProbability = 0;

        if ($stack->shape() !== $prob->shape()) {
            throw new \RuntimeException("Input vector and probability vector has different shapes");
        }

        foreach ($stack as $index => $item) {
            $currentProbability += $prob[$index];
            //$comp = bccomp($currentProbability, $stopAt, 20);
            //echo sprintf("%s >= %s  [%s]\n", $currentProbability, $stopAt, ($comp === 1 || $comp === 0));

            if ($currentProbability >= $stopAt) {
                return $item;
            }
        }
        return null;
    }
}
