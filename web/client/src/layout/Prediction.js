import "antd/dist/antd.css"; // or 'antd/dist/antd.less's
import React, { useState } from "react";

// TODO: create style
function Prediction({ prediction, imageSrc }) {
  console.log(imageSrc);
  return (
    <div>
      <h2
        style={{
          marginLeft: "10px",
          textAlign: "center",
        }}
      >
        Inference Results
      </h2>
      <div>
        {/* image preview */}
        <img
          src={imageSrc}
          alt="preview"
          style={{
            width: "100%",
            height: "auto",
            marginTop: "10px",
          }}
        />
      </div>
    </div>
  );
}

export default Prediction;
