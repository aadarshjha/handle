import "antd/dist/antd.css";
import Webcam from "react-webcam";
import React, { useState } from "react";
import { Form, Button, Radio, Select } from "antd";
const { Option } = Select;

// ideas
// inverting the colors.

const PREFIX = require("../config.json").dev.prefix;

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
  setPrediction,
  setImageSrc,
  setImageOptions,
  imageOptions,
}) {
  const webcamRef = React.useRef(null);

  // for recording video
  const mediaRecorderRef = React.useRef(null);
  const [capturing, setCapturing] = React.useState(false);
  const [recordedChunks, setRecordedChunks] = React.useState([]);

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
    handleDownload();
  }, [mediaRecorderRef, webcamRef, setCapturing]);

  const handleDownload = React.useCallback(() => {
    if (recordedChunks.length) {
      const superBuffer = new Blob(recordedChunks);
      // window.URL.createObjectURL(superBuffer);

      // const blob = new Blob(recordedChunks, {
      //   type: "video/webm",
      // });
      // const url = URL.createObjectURL(blob);
      // const a = document.createElement("a");
      // document.body.appendChild(a);
      // a.style = "display: none";
      // a.href = url;
      // a.download = "capture.webm";
      // a.click();
      // window.URL.revokeObjectURL(url);
      // console.log(recordedChunks);
      setRecordedChunks([]);
    }
  }, [recordedChunks]);

  const updateOptions = (e) => {
    const typeOfButton = e.target.value;

    const typeOfModels = [
      "cnn",
      "densenet",
      "densenet_pretrained",
      "resnet",
      "resnet_pretrained",
      "mobilenet",
      "mobilenet_pretrained",
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

    fetch(`${PREFIX}/static/inference`, {
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
        setImageSrc("data:image/png;base64," + json.image);
        setPrediction(json.prediction);
      });
  }, [webcamRef, imageOptions]);
  return (
    <div style={styles.container}>
      {/* set up a camera frame */}
      <div style={styles.webcam}>
        <Webcam
          style={styles.webcam}
          ref={webcamRef}
          screenshotFormat="image/png"
        />
        {/* if in dynamic mode */}
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
          </div>
        )}
        ;
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
            <Form>
              <Form.Item label="" name="mode">
                <Radio.Group buttonStyle="solid" defaultValue={"cnn"}>
                  <Radio.Button onChange={updateOptions} value="cnn">
                    CNN
                  </Radio.Button>
                  <Radio.Button onChange={updateOptions} value="densenet">
                    DenseNet
                  </Radio.Button>
                  <Radio.Button
                    onChange={updateOptions}
                    value="densenet_pretrained"
                  >
                    DenseNet Pretrained
                  </Radio.Button>
                  <Radio.Button onChange={updateOptions} value="mobilenet">
                    MobileNet
                  </Radio.Button>
                  <Radio.Button
                    onChange={updateOptions}
                    value="mobilenet_pretrained"
                  >
                    MobileNet Pretrained
                  </Radio.Button>
                  <Radio.Button onChange={updateOptions} value="resnet">
                    ResNet
                  </Radio.Button>
                  <Radio.Button
                    onChange={updateOptions}
                    value="resnet_pretrained"
                  >
                    ResNet Pretrained
                  </Radio.Button>
                </Radio.Group>
              </Form.Item>
            </Form>
          </div>
        </div>
      </div>
    </div>
  );
}

export default CameraMenu;
