import { useEffect } from 'react';
import { useParams } from 'react-router-dom';
import './NotebookViewer.css';

const NotebookViewer = () => {
  const { filename } = useParams();
  const src = filename ? `/portfolio/${filename}` : '';

  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  return (
    <div className="iframe-wrapper">
      <iframe className="iframe" src={src} title="Jupyter Notebook" />
    </div>
  );
};

export default NotebookViewer;
