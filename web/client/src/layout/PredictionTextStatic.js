import "antd/dist/antd.css"; // or 'antd/dist/antd.less's
import React, { useState, useEffect } from "react";

function PredictionTextStatic({ prediction, imageSrc, imageOptions }) {
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
        {/* if imageSrc is an empty array, render, otherwise display text */}
        {imageSrc.length === 0 ? (
          <div>
            <h3
              style={{
                marginLeft: "10px",
              }}
            >
              No image preview
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
            <img
              src={imageSrc}
              alt="preview"
              style={{
                width: "320px",
                height: "120px",
                marginTop: "10px",
                marginBottom: "10px",
              }}
            />
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

export default PredictionTextStatic;
