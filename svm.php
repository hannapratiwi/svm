<?php

$modelJson = file_get_contents('svm_model.json');
$modelParams = json_decode($modelJson, true);
function rbfKernel($x, $supportVector, $gamma) {
    $sum = 0;
    for ($i = 0; $i < count($x); $i++) {
        $sum += pow($x[$i] - $supportVector[$i], 2);
    }
    return exp(-$gamma * $sum);
}
function predict($input, $modelParams) {
    $result = 0;
    for ($i = 0; $i < count($modelParams['support_vectors']); $i++) {
        $kernelResult = rbfKernel($input, $modelParams['support_vectors'][$i], $modelParams['gamma']);
        $result += $modelParams['dual_coef'][0][$i] * $kernelResult;
    }
    $result += $modelParams['intercept'][0];
    return $result >= 0 ? 1 : -1;
}

// Font 
$fontFile = fopen("font.csv", 'r');
if ($fontFile) {
    $font = [];
    $ohe_f = [];
    while (($line = fgetcsv($fontFile)) !== false) {
        if($line[0] == "font") continue;
        $font[] = explode(" ",$line[0]);
        $ohe_f[] = 0;
    }
    fclose($fontFile);
}
// print_r($font);

// Warna
$warnaFile = fopen("warna.csv", 'r');
if ($warnaFile) {
    $warna = [];
    $ohe_w = [];
    while (($line = fgetcsv($warnaFile)) !== false) {
        if($line[0] == "warna") continue;
        $warna[] = explode(" ",$line[0]);
        $ohe_w[] = 0;
    }
    fclose($warnaFile);
}
// print_r($warna);

$page = 0;
$menu = 0;
$inputFont = "icomoon icomoon LMRomanSlant10 Raleway Raleway Raleway Raleway Raleway Raleway Raleway Raleway";
$inputColor = "#ff0 #000 #c0c0c0 #fff #111 #fff498 #fff498 #fff498 #fff498 #222 #ddd #e3e3e3 #e3e3e3";

if(isset($_POST['page'])){
    $page = $_POST['page'];
}
if(isset($_POST['menu'])){
    $page = $_POST['menu'];

}

// One Hot Encoding
foreach($font as $key=>$x) {
    foreach($x as $y){
        if(strpos($inputFont,$y) != null){
            $ohe_f[$key] = 1;
            break;
        }
    }
}
// print_r($ohe_f);
foreach($warna as $key=>$x) {
    foreach($x as $y){
        if(strpos($inputColor,$y) != null){
            $ohe_w[$key] = 1;
            break;
        }
    }
}
// print_r($ohe_w);

$inputData = array($page, $menu);
$inputData =  array_merge($inputData, $ohe_f);
$inputData =  array_merge($inputData, $ohe_w);
// print_r($inputData);

$prediction = predict($inputData, $modelParams);

echo "Prediction: " . $prediction;
