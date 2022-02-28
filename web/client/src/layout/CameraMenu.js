import "antd/dist/antd.css";
import Webcam from "react-webcam";
import React, { useState } from "react";
import {
  Form,
  Input,
  Button,
  Radio,
  Select,
  Cascader,
  DatePicker,
  InputNumber,
  TreeSelect,
  Switch,
} from "antd";
const { Option } = Select;

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
    width: "10%",
    marginTop: "10px",
  },
};

function handleChange(value) {
  console.log(`selected ${value}`);
}

function CameraMenu() {
  const webcamRef = React.useRef(null);

  const capture = React.useCallback(() => {
    const imageSrc = webcamRef.current.getScreenshot();
    const new_image = imageSrc.split(",")[1];
    // post imageSrc to http://127.0.0.1:5000/static/cnn
    fetch(`${PREFIX}/static/cnn`, {
      method: "POST",
      mode: "cors",
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        imageSrc: new_image,
      }),
    })
      .then((res) => res.json())
      .then((json) => {
        console.log(json);
      });
  }, [webcamRef]);
  return (
    <div style={styles.container}>
      {/* set up a camera frame */}
      <div style={styles.webcam}>
        <Webcam
          style={styles.webcam}
          ref={webcamRef}
          screenshotFormat="image/png"
        />
        {/* center this button */}
        <Button
          style={styles.capture}
          type="primary"
          size="large"
          onClick={capture}
        >
          Capture
        </Button>
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
                <Radio.Group>
                  <Radio.Button value="static">Static</Radio.Button>
                  <Radio.Button value="dynamic">Dynamic</Radio.Button>
                </Radio.Group>
              </Form.Item>
            </Form>
          </div>

          <div style={{
            marginTop: "-10px",
          }}>
          <h3>Model Mode</h3>
            <Form>
              <Form.Item label="" name="mode">
                <Radio.Group>
                  <Radio.Button value="cnn">CNN</Radio.Button>
                  <Radio.Button value="resnet">ResNet</Radio.Button>
                  <Radio.Button value="densenet">DenseNet</Radio.Button>
                  <Radio.Button value="mobilenet">MobileNet</Radio.Button>
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
