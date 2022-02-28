import "antd/dist/antd.css"; // or 'antd/dist/antd.less'
import CameraMenu from "./CameraMenu";
import React, { useState } from "react";
import Prediction from "./Prediction";

// create style
const styles = {
  container: {
    display: "flex",
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    height: "100vh",
  },
  left: {
    width: "55%",
    height: "100vh",
  },
  right: {
    width: "45%",
    height: "100vh",
  },
};

function Layout() {
  const [prediction, setPrediction] = useState("");

  return (
    <div style={styles.container}>
      <div style={styles.left}>
        <CameraMenu />
      </div>
      <div style={styles.right}>
        <Prediction prediction={prediction} />
      </div>
    </div>
  );
}

export default Layout;
