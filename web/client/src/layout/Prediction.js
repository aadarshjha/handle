import "antd/dist/antd.css"; // or 'antd/dist/antd.less's
import React, { useState, useEffect } from "react";
import PredictionTextStatic from "./PredictionTextStatic";
import PredictionTextDynamic from "./PredictionTextDynamic";

// TODO: create style
function Prediction({ prediction, imageSrc, imageOptions }) {
  console.log(prediction);
  console.log(imageSrc);
  console.log(imageOptions);
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
        prediction={prediction}
        imageSrc={imageSrc}
        imageOptions={imageOptions}
      />
    );
  }
}

export default Prediction;
