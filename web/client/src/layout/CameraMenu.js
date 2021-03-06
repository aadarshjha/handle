import "antd/dist/antd.css";
import Webcam from "react-webcam";
import React, { useEffect, useState } from "react";
import { Form, Button, Radio, Select, Alert } from "antd";

const { Option } = Select;
const PREFIX = require("../config.json").dev.prefix;
const srcInsert = document.getElementById("srcInsert");

const styles = {
  container: {
    display: "flex",
    flexDirection: "column",
    height: "100vh",
    padding: 0,
  },
  webcam: {
    width: "100%",
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    alignItems: "center",
  },
  capture: {
    width: "100%",
    marginTop: "10px",
  },
  videoCapture: {
    width: "100%",
    marginTop: "10px",
  },
};

function CameraMenu({
  setVideoPrediction,
  videoPrediction,
  setPrediction,
  setImageSrc,
  setImageOptions,
  imageOptions,
  geturl,
  seturl,
}) {
  const webcamRef = React.useRef(null);

  // for recording video
  const mediaRecorderRef = React.useRef(null);
  const [capturing, setCapturing] = React.useState(false);
  const [recordedChunks, setRecordedChunks] = React.useState([]);
  const [displayError, setDisplayError] = React.useState(false);

  const handleStartCaptureClick = React.useCallback(() => {
    setCapturing(true);
    mediaRecorderRef.current = new MediaRecorder(webcamRef.current.stream, {
      mimeType: "video/webm",
    });
    mediaRecorderRef.current.addEventListener(
      "dataavailable",
      handleDataAvailable
    );
    mediaRecorderRef.current.start();
  }, [webcamRef, setCapturing, mediaRecorderRef]);

  const handleDataAvailable = React.useCallback(
    ({ data }) => {
      if (data.size > 0) {
        setRecordedChunks((prev) => prev.concat(data));
      }
    },
    [setRecordedChunks]
  );

  const handleStopCaptureClick = React.useCallback(() => {
    mediaRecorderRef.current.stop();
    setCapturing(false);
  }, [mediaRecorderRef, webcamRef, setCapturing]);

  const blobToBase64 = (blob) => {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.readAsDataURL(blob);
      reader.onloadend = function () {
        resolve(reader.result);
      };
    });
  };

  // useEffect

  const handlePreview = React.useCallback(() => {
    if (recordedChunks.length) {
      const blob = new Blob(recordedChunks, {
        type: "video/webm",
      });
      blobToBase64(blob).then((base64) => {
        const jsonString = JSON.stringify({ blob: base64 });
        console.log({videoSrc: jsonString,
          model: imageOptions.model,
          mode: imageOptions.mode})
        // send to our backend.
        fetch(`${PREFIX}/dynamic/cnn`, {
          method: "POST",
          mode: "cors",
          headers: {
            "Access-Control-Allow-Origin": "*",
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            videoSrc: jsonString,
            model: imageOptions.model,
            mode: imageOptions.mode,
          }),
        })
          .then((res) => res.json())
          .then((json) => {
            setVideoPrediction({ ...json });
          });
      });

      seturl(URL.createObjectURL(blob));
      setRecordedChunks([]);
    } else {
      setDisplayError(true);
    }
  }, [recordedChunks, imageOptions, setVideoPrediction, videoPrediction]);

  const updateOptions = (e) => {
    const typeOfButton = e.target.value;

    const typeOfModels = [
      "cnn",
      "densenet_pretrained",
      "vgg_pretrained",
      "mobilenet_pretrained",
      "resnext",
      "lstm",
      "timesformer",
    ];

    // check if typeOfButton is in typeOfModels
    if (typeOfModels.includes(typeOfButton)) {
      setImageOptions({
        ...imageOptions,
        model: typeOfButton,
      });
    } else {
      setImageOptions({
        ...imageOptions,
        mode: typeOfButton,
      });
    }
  };

  // const

  const capture = React.useCallback(() => {
    const imageSrc = webcamRef.current.getScreenshot();
    const new_image = imageSrc.split(",")[1];

    fetch(`${PREFIX}/static/cnn`, {
      method: "POST",
      mode: "cors",
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        imageSrc: new_image,
        model: imageOptions.model,
        mode: imageOptions.mode,
      }),
    })
      .then((res) => res.json())
      .then((json) => {
        setImageSrc({
          HGR: "data:image/png;base64," + json.HGR.image,
          ASL: "data:image/png;base64," + json.ASL.image,
        });
        setPrediction({
          HGR: json.HGR.prediction,
          ASL: json.ASL.prediction,
        });
      });
  }, [webcamRef, imageOptions]);
  return (
    // if displayError is true, then <Alert message="Error Text" type="error" />
    <div style={styles.container}>
      <div>
        {displayError && (
          <Alert
            message="Error"
            description="No video taken!"
            type="error"
            showIcon
            closable
            onClose={() => setDisplayError(false)}
          />
        )}
      </div>

      <div style={styles.webcam}>
        <Webcam
          style={styles.webcam}
          ref={webcamRef}
          screenshotFormat="image/png"
        />
        {imageOptions.mode === "static" ? (
          <div>
            <Button
              style={styles.capture}
              type="primary"
              size="large"
              onClick={capture}
            >
              Capture
            </Button>
          </div>
        ) : !capturing ? (
          <div>
            <Button
              style={styles.videoCapture}
              type="primary"
              size="large"
              onClick={handleStartCaptureClick}
            >
              Start Video
            </Button>
            <Button
              style={styles.videoCapture}
              type="primary"
              size="large"
              onClick={handlePreview}
            >
              Preview
            </Button>
          </div>
        ) : (
          <div>
            <Button
              style={styles.videoCapture}
              type="primary"
              size="large"
              onClick={handleStopCaptureClick}
            >
              Stop Video
            </Button>

            <Button
              style={styles.videoCapture}
              type="primary"
              size="large"
              onClick={handlePreview}
            >
              Preview
            </Button>
          </div>
        )}
      </div>
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          justifyContent: "flex-start",
          height: "100vh",
          padding: 0,
          marginLeft: "20px",
        }}
      >
        <div>
          <h2>Configurable Options</h2>
        </div>
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            justifyContent: "flex-start",
            alignItems: "flex-start",
          }}
        >
          <div>
            <h3>Inferencing Mode</h3>
            <Form>
              <Form.Item label="" name="mode">
                <Radio.Group buttonStyle="solid" defaultValue={"static"}>
                  <Radio.Button onChange={updateOptions} value="static">
                    Static
                  </Radio.Button>
                  <Radio.Button onChange={updateOptions} value="dynamic">
                    Dynamic
                  </Radio.Button>
                </Radio.Group>
              </Form.Item>
            </Form>
          </div>

          <div
            style={{
              marginTop: "-10px",
            }}
          >
            <h3>Model Mode</h3>
            {/* if the mode is static */}
            {imageOptions.mode === "static" ? (
              <Form>
                <Form.Item label="" name="mode">
                  <Radio.Group buttonStyle="solid" defaultValue={"cnn"}>
                    <Radio.Button onChange={updateOptions} value="cnn">
                      CNN
                    </Radio.Button>
                    <Radio.Button
                      onChange={updateOptions}
                      value="densenet_pretrained"
                    >
                      DenseNet Pretrained
                    </Radio.Button>
                    <Radio.Button
                      onChange={updateOptions}
                      value="mobilenet_pretrained"
                    >
                      MobileNet Pretrained
                    </Radio.Button>
                    <Radio.Button
                      onChange={updateOptions}
                      value="vgg_pretrained"
                    >
                      VGG Pretrained
                    </Radio.Button>
                  </Radio.Group>
                </Form.Item>
              </Form>
            ) : (
              <Form>
                <Form.Item label="" name="mode">
                  <Radio.Group buttonStyle="solid" defaultValue={"resnext"}>
                    <Radio.Button onChange={updateOptions} value="resnext">
                      ResNext
                    </Radio.Button>
                    <Radio.Button onChange={updateOptions} value="lstm">
                      LSTM
                    </Radio.Button>
                    <Radio.Button onChange={updateOptions} value="timesformer">
                      TimesFormer
                    </Radio.Button>
                  </Radio.Group>
                </Form.Item>
              </Form>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default CameraMenu;
