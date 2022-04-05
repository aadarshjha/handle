import "antd/dist/antd.css"; // or 'antd/dist/antd.less's
import React, { useState, useEffect } from "react";
import PredictionTextStatic from "./PredictionTextStatic";
import PredictionTextDynamic from "./PredictionTextDynamic";

// TODO: create style
function Prediction({
  videoPrediction,
  prediction,
  imageSrc,
  imageOptions,
  geturl,
}) {
  const [toggleView, changeToggleView] = useState(true);

  useEffect(() => {}, [imageOptions.mode]); // <-- here put the parameter to listen

  if (imageOptions.mode === "static") {
    return (
      <PredictionTextStatic
        prediction={prediction}
        imageSrc={imageSrc}
        imageOptions={imageOptions}
      />
    );
  } else {
    return (
      <PredictionTextDynamic
        videoPrediction={videoPrediction}
        prediction={prediction}
        imageSrc={imageSrc}
        imageOptions={imageOptions}
        geturl={geturl}
      />
    );
  }
}

export default Prediction;
