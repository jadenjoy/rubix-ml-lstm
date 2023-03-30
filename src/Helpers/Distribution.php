<?php

namespace Darvin\LSTM\Helpers;

class Distribution
{
    /**
     * @param float $min
     * @param float $max
     * @param int $mul
     * @return float|int
     */
    public static function uniform(float $min = 0, float $max = 1, int $mul = 100000000000000): float|int
    {
        if ($min > $max) throw new \RuntimeException("Min value cannot be greater than max.");
        return mt_rand($min * $mul, $max * $mul) / $mul;
    }

}