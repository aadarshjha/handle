import "antd/dist/antd.css"; // or 'antd/dist/antd.less's
import React, { useState, useEffect } from "react";

function PredictionTextDynamic({ prediction, imageSrc, imageOptions, geturl }) {
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

      <p
        style={{
          marginLeft: "10px",
          textAlign: "center",
        }}
      >
        Deploying {imageOptions.model.toUpperCase()} in{" "}
        {imageOptions.mode.toUpperCase()} mode.
      </p>
      <div>
        {/* if url is an empty string, don't render anything */}
        {geturl === "" ? (
          <div>
            <h3
              style={{
                marginLeft: "10px",
              }}
            >
              No video preview
            </h3>
          </div>
        ) : (
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              justifyContent: "center",
              alignItems: "center",
            }}
          >
            <video style={{ width: "70%" }} controls src={geturl}></video>
          </div>
        )}
      </div>

      <div>
        {/* if prediction is an empty array, render, otherwise display text */}
        {prediction.length === 0 ? (
          <div>
            <h3
              style={{
                marginLeft: "10px",
              }}
            >
              No prediction
            </h3>
          </div>
        ) : (
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              justifyContent: "center",
              alignItems: "flex-start",
            }}
          >
            <h3
              style={{
                marginLeft: "10px",
              }}
            >
              Hand Gesture Recognition Database
            </h3>
            <p
              style={{
                marginLeft: "10px",
              }}
            >
              {prediction}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default PredictionTextDynamic;
