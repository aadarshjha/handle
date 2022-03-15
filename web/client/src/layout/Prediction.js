import "antd/dist/antd.css"; // or 'antd/dist/antd.less's
import React, { useState, useEffect } from "react";
import PredictionText from "./PredictionText";

// TODO: create style
function Prediction({ prediction, imageSrc, imageOptions }) {
  const [toggleView, changeToggleView] = useState(true);

  useEffect(() => {}, [imageOptions.mode]); // <-- here put the parameter to listen

  return (
    <PredictionText
      prediction={prediction}
      imageSrc={imageSrc}
      imageOptions={imageOptions}
    />
  );
}

export default Prediction;
