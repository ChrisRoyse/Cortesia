import React from 'react';
import { Card, Row, Col, Progress, Statistic, Alert } from 'antd';
import { ThunderboltOutlined, StopOutlined } from '@ant-design/icons';
import { InhibitionExcitationBalance as IEBalance } from '../../types/cognitive';

interface InhibitionExcitationBalanceProps {
  balance: IEBalance;
}

export const InhibitionExcitationBalance: React.FC<InhibitionExcitationBalanceProps> = ({ balance }) => {
  const isBalanceOptimal = balance.balance >= balance.optimalRange[0] && balance.balance <= balance.optimalRange[1];
  
  const getBalanceColor = () => {
    if (isBalanceOptimal) return '#52c41a';
    if (balance.balance < balance.optimalRange[0]) return '#faad14';
    return '#ff4d4f';
  };

  const getBalanceText = () => {
    if (balance.balance > 0.5) return 'Excitation Dominant';
    if (balance.balance < -0.5) return 'Inhibition Dominant';
    return 'Balanced';
  };

  const balancePercentage = ((balance.balance + 1) / 2) * 100; // Convert -1 to 1 range to 0 to 100

  return (
    <div>
      {/* Overall Balance */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} md={8}>
          <Card>
            <Statistic
              title="Current Balance"
              value={balance.balance.toFixed(3)}
              valueStyle={{ color: getBalanceColor() }}
            />
            <Progress
              percent={balancePercentage}
              strokeColor={getBalanceColor()}
              trailColor="#f0f0f0"
              style={{ marginTop: 8 }}
            />
            <div style={{ marginTop: 8, fontSize: '12px', textAlign: 'center' }}>
              {getBalanceText()}
            </div>
          </Card>
        </Col>
        <Col xs={24} md={8}>
          <Card>
            <Statistic
              title="Total Excitation"
              value={(balance.excitation.total * 100).toFixed(1)}
              suffix="%"
              prefix={<ThunderboltOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col xs={24} md={8}>
          <Card>
            <Statistic
              title="Total Inhibition"
              value={(balance.inhibition.total * 100).toFixed(1)}
              suffix="%"
              prefix={<StopOutlined />}
              valueStyle={{ color: '#ff4d4f' }}
            />
          </Card>
        </Col>
      </Row>

      {/* Balance Status Alert */}
      {!isBalanceOptimal && (
        <Alert
          message={`Balance outside optimal range (${balance.optimalRange[0].toFixed(2)} to ${balance.optimalRange[1].toFixed(2)})`}
          description={
            balance.balance < balance.optimalRange[0] 
              ? "System is experiencing excessive inhibition. Consider increasing excitatory signals."
              : "System is experiencing excessive excitation. Consider increasing inhibitory controls."
          }
          type="warning"
          showIcon
          style={{ marginBottom: 24 }}
        />
      )}

      {/* Regional Analysis */}
      <Row gutter={[16, 16]}>
        <Col xs={24} lg={12}>
          <Card title="Excitation by Region">
            <div>
              {Object.entries(balance.excitation.byRegion).map(([region, level]) => (
                <div key={region} style={{ marginBottom: 16 }}>
                  <div style={{ marginBottom: 4 }}>
                    <span style={{ textTransform: 'capitalize', fontWeight: 500 }}>
                      {region}
                    </span>
                    <span style={{ float: 'right' }}>
                      {(level * 100).toFixed(1)}%
                    </span>
                  </div>
                  <Progress
                    percent={level * 100}
                    strokeColor="#1890ff"
                    size="small"
                  />
                </div>
              ))}
            </div>
            <div style={{ marginTop: 16, fontSize: '12px' }}>
              <strong>Active Patterns:</strong>
              <div style={{ marginTop: 4 }}>
                {balance.excitation.patterns.map((patternId, index) => (
                  <span key={patternId}>
                    {patternId}
                    {index < balance.excitation.patterns.length - 1 && ', '}
                  </span>
                ))}
              </div>
            </div>
          </Card>
        </Col>
        
        <Col xs={24} lg={12}>
          <Card title="Inhibition by Region">
            <div>
              {Object.entries(balance.inhibition.byRegion).map(([region, level]) => (
                <div key={region} style={{ marginBottom: 16 }}>
                  <div style={{ marginBottom: 4 }}>
                    <span style={{ textTransform: 'capitalize', fontWeight: 500 }}>
                      {region}
                    </span>
                    <span style={{ float: 'right' }}>
                      {(level * 100).toFixed(1)}%
                    </span>
                  </div>
                  <Progress
                    percent={level * 100}
                    strokeColor="#ff4d4f"
                    size="small"
                  />
                </div>
              ))}
            </div>
            <div style={{ marginTop: 16, fontSize: '12px' }}>
              <strong>Active Patterns:</strong>
              <div style={{ marginTop: 4 }}>
                {balance.inhibition.patterns.map((patternId, index) => (
                  <span key={patternId}>
                    {patternId}
                    {index < balance.inhibition.patterns.length - 1 && ', '}
                  </span>
                ))}
              </div>
            </div>
          </Card>
        </Col>
      </Row>

      {/* Balance Timeline (placeholder for future implementation) */}
      <Card title="Balance Over Time" style={{ marginTop: 16 }}>
        <div style={{ height: '200px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#666' }}>
          Timeline visualization will be implemented here
        </div>
      </Card>
    </div>
  );
};