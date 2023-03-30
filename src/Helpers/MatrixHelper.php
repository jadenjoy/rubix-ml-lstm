<?php
namespace Darvin\LSTM\Helpers;

use Tensor\Exceptions\RuntimeException;
use Tensor\Matrix;
use Tensor\Vector;

class MatrixHelper extends Matrix
{
    /**
     * @param Matrix $matrix
     * @return Matrix
     */
    public static function zerosLike(Matrix $matrix): Matrix
    {
        return Matrix::zeros($matrix->m(), $matrix->n());
    }

    /**
     * @param Matrix $matrix
     * @return Matrix
     */
    public static function onesLike(Matrix $matrix): Matrix
    {
        return Matrix::ones($matrix->m(), $matrix->n());
    }

    /**
     * @param $index
     * @param Matrix $matrix
     * @param VectorHelper $vector
     * @return void
     */
    public static function setVectorAt($index, Matrix &$matrix, Vector $vector)
    {
        $original = $matrix->offsetGet($index);
        if($original->shape() === $vector->shape()) {
            $mo = $matrix->asArray();
            $mo[$index] = $vector->asArray();
            $matrix = parent::quick($mo);
        } else {
            throw new RuntimeException('Vectors has different shapes.');
        }
    }

    /**
     * @param int $index
     * @param Matrix $matrix
     * @param $value
     */
    public static function setAsFlatten(int $index, Matrix &$matrix, $value)
    {
        $mFlat = $matrix->flatten();
        VectorHelper::setAt($index, $mFlat, $value);
        $matrix = $mFlat->reshape($matrix->m(), $matrix->n());
    }

    /**
     * @param Matrix ...$matrices
     */
    public static function vstack(Matrix ...$matrices): Matrix
    {
        $shapeN = null;
        $resultArr = [];
        foreach ($matrices as $matrix) {
            if(is_null($shapeN)) {
                $shapeN = $matrix->n();
            }
            if ($matrix->n() === $shapeN) {
                $resultArr = array_merge($resultArr, $matrix->asArray());
            } else {
                throw new RuntimeException('Matrices has different shapes.');
            }
        }
        return parent::quick($resultArr);
    }

    /**
     * @param Matrix $matrix
     * @param int $offset
     * @param int|null $length
     * @return Matrix
     */
    public static function slice(Matrix $matrix, int $offset, ?int $length = null): Matrix
    {
        return parent::quick(array_slice($matrix->asArray(), $offset, $length));
    }


}


