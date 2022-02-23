
import 'antd/dist/antd.css'; 
import Webcam from "react-webcam";
import React, { useState } from 'react';
import { Button } from 'antd';


import { Select } from 'antd';
const { Option } = Select;


const styles = {
    container: {
        display: 'flex',
        flexDirection: 'column',
        height: '100vh',
        padding: 0
    },
    webcam: {
        width: '100%', 
    }

}

function handleChange(value) {
    console.log(`selected ${value}`);
}

function CameraMenu() {
    const webcamRef = React.useRef(null);

    const capture = React.useCallback(
      () => {
        const imageSrc = webcamRef.current.getScreenshot();

        // send a POST request to the server with the image data
        fetch("/api/upload", {
            method: "POST",
            body: imageSrc,
        })
        .then(res => res.json())
        .then(data => {
            console.log(data);
            setURL(data.url);
        })
        .catch(err => {
            console.log(err);
        });
      },
      [webcamRef]
    );
  return (
    <div style={styles.container}>
        {/* set up a camera frame */}
        <div style={styles.webcam}>
            <Webcam 
                style={styles.webcam}
                ref={webcamRef}
                screenshotFormat="image/jpeg"
            />
            <Button onClick={capture}>Capture photo</Button>
        </div>
        <div style={{
            display: 'flex',
            flexDirection: 'row',
            justifyContent: 'space-around',
            alignItems: 'center',
            height: '100vh',
            padding: 0
        }}>
            <div>
                <h3>Static | Dynamic Mode</h3>
                <Select style={{ width: 120 }} onChange={handleChange}>
                    <Option value="static">Static</Option>
                    <Option value="dynamic">Dynamic</Option>
                </Select>
            </div>

            {/* this changes */}
            <div>
                <h3>Network Mode</h3>
                <Select style={{ width: 120 }} onChange={handleChange}>
                    <Option value="cnn">CNN</Option>
                    <Option value="resnet">ResNet</Option>
                    <Option value="densenet">DenseNet</Option>
                    <Option value="mobilenet">MobileNet</Option>
                </Select>
            </div>
        </div>
    </div>
  );
}

export default CameraMenu;
