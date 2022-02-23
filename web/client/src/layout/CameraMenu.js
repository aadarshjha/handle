
import 'antd/dist/antd.css'; 
import Webcam from "react-webcam";

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

function CameraMenu() {
  return (
    <div style={styles.container}>
        {/* set up a camera frame */}
        <div style={styles.webcam}>
            <Webcam style={styles.webcam}/>
        </div>
        <div>
            <p>Hello world</p>
        </div>
    </div>
  );
}

export default CameraMenu;
