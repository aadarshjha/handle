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
              }}
            />
          </div>
        )}
      </div>
    </div>
  );
}

export default Prediction;
