<?php
namespace Darvin\LSTM\Helpers;

use Tensor\Vector;

class VectorHelper extends Vector {

    /**
     * @param int $index
     * @param Vector $vector
     * @param $value
     */
    public static function setAt(int $index, Vector &$vector, $value) {
        $vector->offsetGet($index);
        $arr = $vector->asArray();
        $arr[$index] = $value;
        $vector = Vector::quick($arr);
    }
}