import React, { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const MarkdownRenderer = ({ file }) => {
  const [content, setContent] = useState('');
  const [error, setError] = useState(null);

  useEffect(() => {
    const filePath = `/blogs/markdowns/${file}`;
    console.log(`Attempting to fetch: ${filePath}`);
    fetch(filePath)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Network response was not ok: ${response.statusText}`);
        }
        
        return response.text();
      })
      .then((text) => {
        console.log('Fetched content:', text);
        setContent(text);
      })
      .catch((error) => {
        console.error('Error fetching markdown file:', error);
        setError(error.message);
      });
  }, [file]);

  if (error) {
    return <div>Error: {error}</div>;
  }

  return (
    <div className="markdown-body">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>
        {content}
      </ReactMarkdown>
    </div>
  );
};

export default MarkdownRenderer;
