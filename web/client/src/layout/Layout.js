
import 'antd/dist/antd.css'; // or 'antd/dist/antd.less'
import CameraMenu from './CameraMenu';

// create style 
const styles = {
    container: {
        display: 'flex',
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center', 
        height: '100vh',
    },
    left: {
        width: '60%',
        height: '100vh',
        backgroundColor: 'red',
    },
    right: {
        width: '40%',
        height: '100vh',
        backgroundColor: 'blue',
    },
}

function Layout() {
  return (
    <div style={styles.container}>
        <div style={styles.left}>
            <CameraMenu />
        </div>
        <div style={styles.right}>
        </div>
    </div>
  );
}

export default Layout;
